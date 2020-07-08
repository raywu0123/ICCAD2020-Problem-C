#include <queue>

#include "simulator/simulator.h"
#include "simulator/collision_utils.h"
#include "include/progress_bar.h"

using namespace std;


__device__ __host__ void resolve_collisions_for_batch_stimuli(
    Transition** data,
    const unsigned int* lengths,
    const unsigned int* capacities,
    const unsigned int num_outputs, const unsigned int* output_indices
) {
    auto* stimuli_lengths = new unsigned int[INITIAL_CAPACITY];
    for (int i_output = 0; i_output < num_outputs; i_output++) {
        for(int i_stimuli = 0; i_stimuli < N_STIMULI_PARALLEL; i_stimuli++) {
            stimuli_lengths[i_stimuli] = lengths[num_outputs * i_stimuli + i_output];
        }
        resolve_collisions_for_batch_waveform(
            data[output_indices[i_output]], capacities[output_indices[i_output]],
            stimuli_lengths
        );
    }
    delete[] output_indices;
    delete[] stimuli_lengths;
}

__device__ void init_delay_info(
    Transition** data, const unsigned int* data_schedule_args, unsigned int data_schedule_size, const unsigned int* capacities,
    unsigned int num_input
) {
    for (unsigned int i = 0; i < data_schedule_size; i++) {
        const auto& arg = data_schedule_args[i];
        if (arg >= num_input) continue;
        if (data[i][0].delay_info.edge_type == 1) continue; // use first edge type as initialize flag

        for (unsigned int j = 1; j < capacities[i]; j++) {
            if (data[i][j].value == 0) break;
            data[i][j].delay_info.arg = arg;
            data[i][j].delay_info.edge_type = get_edge_type(data[i][j - 1].value, data[i][j].value);
        }
    }
}

__device__ void simulate_module(
    const ModuleSpec* const module_spec,
    const SDFSpec* const sdf_spec,
    Data* const data_schedule,
    bool* const overflow
) {
    auto* data_ptrs_for_each_stimuli = new Transition*[module_spec->data_schedule_size];
    auto* capacities = new unsigned int[module_spec->data_schedule_size];
    unsigned stimuli_idx = threadIdx.x;
    for (unsigned int i = 0; i < module_spec->data_schedule_size; i++) {
        data_ptrs_for_each_stimuli[i] = data_schedule[i].ptr + stimuli_idx * data_schedule[i].capacity;
        capacities[i] = data_schedule[i].capacity;
    }

    init_delay_info(
        data_ptrs_for_each_stimuli, module_spec->data_schedule_args, module_spec->data_schedule_size, capacities,
        module_spec->num_module_input
    );
    unsigned int data_schedule_idx = 0;
    for (int i = 0; i < module_spec->schedule_size; i++) {
        module_spec->gate_schedule[i](
            data_ptrs_for_each_stimuli + data_schedule_idx,
            capacities + data_schedule_idx,
            module_spec->tables[i], module_spec->table_row_num[i],
            module_spec->num_inputs[i], module_spec->num_outputs[i],
            overflow
        );
        data_schedule_idx += module_spec->num_inputs[i] + module_spec->num_outputs[i];
    }

    __shared__ unsigned int lengths[N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT];
    compute_delay(
        data_ptrs_for_each_stimuli, capacities,
        module_spec->output_indices, module_spec->num_module_output, module_spec->num_module_input,
        sdf_spec, lengths + stimuli_idx * module_spec->num_module_output
    );

    __syncthreads();
    if (threadIdx.x == 0) {
        resolve_collisions_for_batch_stimuli(
            data_ptrs_for_each_stimuli, lengths, capacities,
            module_spec->num_module_output, module_spec->output_indices
        );
    }

    delete[] data_ptrs_for_each_stimuli;
    delete[] capacities;
}

__global__ void simulate_batch(BatchResource batch_resource) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& offset = batch_resource.data_schedule_offsets[blockIdx.x];
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& sdf_spec = batch_resource.sdf_specs[blockIdx.x];
        auto module_data_schedule = &batch_resource.data_schedule[offset];
        auto* overflow = batch_resource.overflows[blockIdx.x];
        simulate_module(module_spec, sdf_spec, module_data_schedule, overflow);
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
        queue<Cell*, deque<Cell*>> job_queue(deque<Cell*>(schedule_layer.begin(), schedule_layer.end()));
        int session_id = 0;

        while (not job_queue.empty()) {
            unordered_set<Cell*> processing_cells;

            for (int i = 0; i < N_GATE_PARALLEL; i++) {
                auto* cell = job_queue.front(); job_queue.pop(); processing_cells.insert(cell);
                cell->prepare_resource(session_id, resource_buffer);
                if (not cell->finished()) job_queue.push(cell);
                if (job_queue.empty()) break;
            }
            const auto& batch_data = BatchResource{resource_buffer};
            resource_buffer.clear();
            simulate_batch<<<N_GATE_PARALLEL, N_STIMULI_PARALLEL>>>(batch_data);
            cudaDeviceSynchronize();

            for (auto* cell : processing_cells) {
                if (cell->overflow()) {
                    if (cell->finished()) job_queue.push(cell);
                    cell->handle_overflow();
                } else cell->dump_result();
            }
            session_id++;
        }
        progress_bar.Progressed(i_layer + 1);
    }
    cout << endl;
}
