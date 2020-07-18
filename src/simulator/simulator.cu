#include <stack>
#include <cassert>

#include "simulator/simulator.h"
#include "simulator/collision_utils.h"
#include "include/progress_bar.h"

using namespace std;


__device__ __host__ void resolve_collisions_for_batch_stimuli(
    Transition** data,
    const unsigned int* lengths,
    const unsigned int capacity,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
//    TODO parallelize
    unsigned int stimuli_lengths[N_STIMULI_PARALLEL];
    for (int i_output = 0; i_output < num_outputs; i_output++) {
        for(int i_stimuli = 0; i_stimuli < N_STIMULI_PARALLEL; i_stimuli++) {
            stimuli_lengths[i_stimuli] = lengths[num_outputs * i_stimuli + i_output];
            assert(stimuli_lengths[i_stimuli] <= capacity);
        }
        resolve_collisions_for_batch_waveform(
            data[num_inputs + i_output], capacity,
            stimuli_lengths, N_STIMULI_PARALLEL
        );
    }
}

__device__ void init_delay_info(Transition** data, unsigned num_input) {
    for (unsigned int i = 0; i < num_input; i++) {
        for (unsigned int j = 1; j < INITIAL_CAPACITY; j++) {
             if (data[i][j].value == 0) break;
             data[i][j].delay_info.arg = i;
             data[i][j].delay_info.edge_type = get_edge_type(data[i][j - 1].value, data[i][j].value);
        }
    }
}

__device__ void simulate_module(
    const ModuleSpec* const module_spec,
    const SDFSpec* const sdf_spec,
    Transition** const data,
    const unsigned int capacity,
    bool* overflow_ptr
) {
    unsigned stimuli_idx = threadIdx.x;
    Transition* data_ptrs_for_each_stimuli[MAX_NUM_MODULE_ARGS];
    for (unsigned int i = 0; i < module_spec->num_module_args; i++) {
        data_ptrs_for_each_stimuli[i] = data[i] + stimuli_idx * capacity;
    }
    init_delay_info(data_ptrs_for_each_stimuli, module_spec->num_module_input);

    unsigned int offset = 0;
    for (int i = 0; i < module_spec->schedule_size; i++) {
        const unsigned int num_gate_args = module_spec->num_inputs[i] + module_spec->num_outputs[i];
        assert(num_gate_args <= MAX_NUM_GATE_ARGS);
        Transition* data_schedule_for_gate[MAX_NUM_GATE_ARGS] = { nullptr };
        for (int j = 0; j < num_gate_args; ++j) {
            const auto& arg = module_spec->gate_specs[offset + j];
            data_schedule_for_gate[j] = data_ptrs_for_each_stimuli[arg];
        }
        module_spec->gate_schedule[i](
            data_schedule_for_gate,
            capacity,
            module_spec->tables[i], module_spec->table_row_num[i],
            module_spec->num_inputs[i], module_spec->num_outputs[i],
            overflow_ptr
        );
        offset += num_gate_args;
    }
    assert(module_spec->num_module_output <= MAX_NUM_MODULE_OUTPUT);
    __shared__ unsigned int lengths[N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT];
    compute_delay(
        data_ptrs_for_each_stimuli, capacity,
        module_spec->num_module_output, module_spec->num_module_input,
        sdf_spec, lengths + stimuli_idx * module_spec->num_module_output
    );

    __syncthreads();
    if (threadIdx.x == 0) {
        resolve_collisions_for_batch_stimuli(
            data, lengths, capacity,
            module_spec->num_module_input, module_spec->num_module_output
        );
    }
}

__global__ void simulate_batch(BatchResource batch_resource) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& offset = batch_resource.data_schedule_offsets[blockIdx.x];
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& sdf_spec = batch_resource.sdf_specs[blockIdx.x];
        auto* module_data = &batch_resource.data_schedule[offset];
        const auto& capacity = batch_resource.capacities[blockIdx.x];
        auto* overflow_ptr = batch_resource.overflows[blockIdx.x];
        simulate_module(module_spec, sdf_spec, module_data, capacity, overflow_ptr);
    }
}

void Simulator::run() {
    cout << "| Status: Running Simulation... " << endl;

    unsigned int num_layers = circuit.cell_schedule.size();
    cout << "| Total " << num_layers << " layers" << endl;

    ProgressBar progress_bar(num_layers);
    for (unsigned int i_layer = 0; i_layer < num_layers; i_layer++) {
        const auto& schedule_layer = circuit.cell_schedule[i_layer];
        for (auto* cell : schedule_layer) cell->init();
        stack<Cell*, std::vector<Cell*>> job_queue(schedule_layer);
        int session_id = 0;

        while (not job_queue.empty()) {
            unordered_set<Cell*> processing_cells;
            ResourceBuffer resource_buffer;
            for (int i = 0; i < N_CELL_PARALLEL; i++) {
                if (job_queue.empty()) break;
                auto* cell = job_queue.top(); processing_cells.insert(cell);
                cell->prepare_resource(session_id, resource_buffer);
                if (cell->finished()) job_queue.pop();
            }
            BatchResource batch_data{}; batch_data.init(resource_buffer);
            simulate_batch<<<N_CELL_PARALLEL, N_STIMULI_PARALLEL>>>(batch_data);
            cudaDeviceSynchronize();

            for (auto* cell : processing_cells) {
                if (cell->overflow()) {
                    if (cell->finished()) job_queue.push(cell);
                    cell->handle_overflow();
                }
                else cell->dump_result();
            }
            batch_data.free();
            session_id++;
        }
        progress_bar.Progressed(i_layer + 1);
    }
    cout << endl;
}
