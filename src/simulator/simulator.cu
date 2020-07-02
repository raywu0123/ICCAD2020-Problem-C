#include <queue>

#include "simulator/simulator.h"
#include "simulator/collision_utils.h"
#include "include/progress_bar.h"

using namespace std;


__device__ __host__ void resolve_collisions_for_single_stimuli(
    Transition** data,
    unsigned int* lengths,
    const unsigned int* capacities,
    const unsigned int num_outputs, const unsigned int* output_indices
) {
    for (int i = 0; i < num_outputs; i++) {
        resolve_collisions_for_single_waveform(
            data[output_indices[i]], capacities[output_indices[i]], lengths + i
        );
    }
    delete[] output_indices;
}

__device__ __host__ void resolve_collisions_for_batch_stimuli(
    Transition** data,
    unsigned int** batch_lengths,
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
            stimuli_lengths, batch_lengths[output_indices[i_output]]
        );
    }
    delete[] output_indices;
    delete[] stimuli_lengths;
}

__device__ void resolve_collisions(
    Transition** data, const unsigned int* capacities, unsigned int** batch_lengths,
    const unsigned int* output_indices, const unsigned int num_module_output
) {
    __shared__ unsigned int lengths[N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT];

    unsigned int stimuli_idx = threadIdx.x;
    resolve_collisions_for_single_stimuli(
        data, lengths + stimuli_idx * num_module_output, capacities,
        num_module_output, output_indices
    );
    __syncthreads();
    if (threadIdx.x == 0) {
        resolve_collisions_for_batch_stimuli(
            data, batch_lengths, lengths, capacities,
            num_module_output, output_indices
        );
    }
}

__device__ void simulate_gate(
    GateFnPtr gate_fn_ptr,
    Transition** data,  //(capacities[i_wire], num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table,
    const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow
) {
    gate_fn_ptr(data, capacities, table, table_row_num, num_inputs, num_outputs, overflow);
}

__device__ char get_edge_type(char v1, char v2) {
    if (v2 == '1' or v1 == '0') return '+';
    if (v2 == '0' or v1 == '1') return '-';
    return 'x';
}

__device__ void init_delay_info(
    Transition** data, const unsigned int* data_schedule_args, unsigned int data_schedule_size, const unsigned int* capacities,
    unsigned int num_input
) {
    for (unsigned int i = 0; i < data_schedule_size; i++) {
        const auto& arg = data_schedule_args[i];
        if (arg >= num_input) continue;
        if (data[i][0].delay_info.edge_type == 1) continue; // use first edge type as initialize flag

        data[i][0].delay_info.edge_type = 1;
        for (unsigned int j = 1; j < capacities[i]; j++) {
            if (data[i][j].value == 0) break;
            data[i][j].delay_info.arg = arg; data[i][j].delay_info.edge_type = get_edge_type(data[i][j - 1].value, data[i][j].value);
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
    auto** batch_lengths = new unsigned int*[module_spec->data_schedule_size];
    unsigned stimuli_idx = threadIdx.x;
    for (unsigned int i = 0; i < module_spec->data_schedule_size; i++) {
        data_ptrs_for_each_stimuli[i] = data_schedule[i].ptr + stimuli_idx * data_schedule[i].capacity;
        capacities[i] = data_schedule[i].capacity;
        batch_lengths[i] = &(data_schedule[i].length);
    }

    init_delay_info(
        data_ptrs_for_each_stimuli, module_spec->data_schedule_args, module_spec->data_schedule_size, capacities,
        module_spec->num_module_input
    );
    unsigned int data_schedule_idx = 0;
    for (int i = 0; i < module_spec->schedule_size; i++) {
        simulate_gate(
            module_spec->gate_schedule[i],
            data_ptrs_for_each_stimuli + data_schedule_idx,
            capacities + data_schedule_idx,
            module_spec->tables[i],
            module_spec->table_row_num[i],
            module_spec->num_inputs[i], module_spec->num_outputs[i],
            overflow
        );
        data_schedule_idx += module_spec->num_inputs[i] + module_spec->num_outputs[i];
    }
    compute_delay(
        data_ptrs_for_each_stimuli, capacities,
        module_spec->output_indices, module_spec->num_module_output, module_spec->num_module_input,
        sdf_spec
    );
    resolve_collisions(
        data_ptrs_for_each_stimuli, capacities, batch_lengths,
        module_spec->output_indices, module_spec->num_module_output
    );

    delete[] data_ptrs_for_each_stimuli;
    delete[] capacities;
    delete[] batch_lengths;
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
    cout << "Running Simulation... " << endl;

    unsigned int num_layers = circuit.cell_schedule.size();
    cout << "Total " << num_layers << " layers" << endl;

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
