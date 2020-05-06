#include "simulator/simulator.h"
#include "include/progress_bar.h"

using namespace std;


extern void __host__ __device__ get_output_indices(
    unsigned int* output_indices,
    unsigned int* data_schedule_indices, unsigned int data_schedule_size,
    unsigned int num_inputs, unsigned int num_outputs
) {
    for (int i = 0; i < data_schedule_size; i++) {
        if (num_inputs <= data_schedule_indices[i] and data_schedule_indices[i] < num_inputs + num_outputs) {
            output_indices[data_schedule_indices[i] - num_inputs] = i;
        }
    }
}

__device__ void compute_delay_on_multiple_stimuli(
    Data* data,
    const ModuleSpec* module_spec,
    const SDFSpec* sdf_spec
) {
    unsigned int stimuli_idx = threadIdx.x;
    const auto& data_schedule_size = module_spec->data_schedule_size;
    auto** stimuli_data = new Transition*[data_schedule_size]; // (capacities[i], num_inputs + num_outputs)
    auto* capacities = new unsigned int[data_schedule_size];
    for (int i = 0; i < data_schedule_size; i++) {
        stimuli_data[i] = data[i].ptr + data[i].capacity * stimuli_idx;
        capacities[i] = data[i].capacity;
    }
    compute_delay(
        stimuli_data,
        data_schedule_size,
        capacities,
        module_spec->data_schedule_indices,
        module_spec->num_module_input, module_spec->num_module_output,
        sdf_spec
    );
    delete[] stimuli_data;
    delete[] capacities;
}

__device__ __host__ void resolve_collisions_for_single_stimuli(
    Transition** data,
    unsigned int* lengths,
    unsigned int data_schedule_size,
    unsigned int* capacities,
    unsigned int* data_schedule_indices,
    unsigned int num_inputs, unsigned int num_outputs
) {
    auto* output_indices = new unsigned int[num_outputs];
    get_output_indices(output_indices, data_schedule_indices, data_schedule_size, num_inputs, num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        resolve_collisions_for_single_waveform(
                data[output_indices[i]], capacities[output_indices[i]], lengths + i
        );
    }
    delete[] output_indices;
}

__device__ __host__ void resolve_collisions_for_batch_stimuli(
    Transition** data,
    unsigned int ** batch_lengths,
    unsigned int* lengths,
    unsigned int data_schedule_size,
    unsigned int* capacities,
    unsigned int* data_schedule_indices,
    unsigned int num_inputs, unsigned int num_outputs
) {
    auto* output_indices = new unsigned int[num_outputs];
    auto* stimuli_lengths = new unsigned int[INITIAL_CAPACITY];

    get_output_indices(output_indices, data_schedule_indices, data_schedule_size, num_inputs, num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        for(int i_stimuli = 0; i_stimuli < N_STIMULI_PARALLEL; i_stimuli++) {
            stimuli_lengths[i] = lengths[num_outputs * i_stimuli + i];
        }
        resolve_collisions_for_batch_waveform(
                data[output_indices[i]], capacities[output_indices[i]], stimuli_lengths,
                batch_lengths[output_indices[i]]
        );

    }
    delete[] output_indices;
    delete[] stimuli_lengths;
}

__device__ void resolve_collisions_on_multiple_stimuli(
    Data* data,
    const ModuleSpec* module_spec
) {
    unsigned int stimuli_idx = threadIdx.x;
    const auto& data_schedule_size = module_spec->data_schedule_size;

    auto* lengths = new unsigned int[N_STIMULI_PARALLEL * module_spec->num_module_output];

    auto** stimuli_data = new Transition*[data_schedule_size]; // (capacities[i], num_inputs + num_outputs)
    auto* capacities = new unsigned int[data_schedule_size];
    auto** batch_lengths = new unsigned int*[data_schedule_size];
    for (int i = 0; i < data_schedule_size; i++) {
        stimuli_data[i] = data[i].ptr + data[i].capacity * stimuli_idx;
        capacities[i] = data[i].capacity;
        batch_lengths[i] = &data[i].length;
    }

    resolve_collisions_for_single_stimuli(
            stimuli_data,
            lengths + stimuli_idx * module_spec->num_module_output,
            data_schedule_size,
            capacities,
            module_spec->data_schedule_indices,
            module_spec->num_module_input, module_spec->num_module_output
    );
    __syncthreads();
    if (threadIdx.x == 0) {
        resolve_collisions_for_batch_stimuli(
                stimuli_data,
                batch_lengths,
                lengths,
                data_schedule_size,
                capacities,
                module_spec->data_schedule_indices,
                module_spec->num_module_input, module_spec->num_module_output
        );
    }
    delete[] lengths;
    delete[] batch_lengths;
    delete[] stimuli_data;
    delete[] capacities;
}

__device__ void simulate_gate_on_multiple_stimuli(
        GateFnPtr gate_fn_ptr,
        Data* data,  //(n_stimuli * capacities[i_wire], num_inputs + num_outputs)
        char* table,
        unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    unsigned int stimuli_idx = threadIdx.x;

    auto** stimuli_data = new Transition*[num_inputs + num_outputs]; // (capacities[i], num_inputs + num_outputs)
    auto* capacities  = new unsigned int[num_inputs + num_outputs];
    for (int i = 0; i < num_inputs + num_outputs; i++) {
        stimuli_data[i] = data[i].ptr + data[i].capacity * stimuli_idx;
        capacities[i] = data[i].capacity;
    }
    gate_fn_ptr(stimuli_data, capacities, table, table_row_num, num_inputs, num_outputs);

    delete[] stimuli_data;
    delete[] capacities;
}

__device__ void simulate_module(
    const ModuleSpec* module_spec,
    const SDFSpec* sdf_spec,
    Data* data_schedule
) {
    unsigned int data_schedule_idx = 0;
    for (int i = 0; i < module_spec->schedule_size; i++) {
        simulate_gate_on_multiple_stimuli(
            module_spec->gate_schedule[i],
            data_schedule + data_schedule_idx,
            module_spec->tables[i],
            module_spec->table_row_num[i],
            module_spec->num_inputs[i], module_spec->num_outputs[i]
        );
        data_schedule_idx += module_spec->num_inputs[i] + module_spec->num_outputs[i];
    }
    compute_delay_on_multiple_stimuli(data_schedule, module_spec, sdf_spec);
    resolve_collisions_on_multiple_stimuli(data_schedule, module_spec);
}

__global__ void simulate_batch(BatchResource batch_resource) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& offset = batch_resource.data_schedule_offsets[blockIdx.x];
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& sdf_spec = batch_resource.sdf_specs[blockIdx.x];
        auto module_data_schedule = &batch_resource.data_schedule[offset];
        simulate_module(module_spec, sdf_spec, module_data_schedule);
    }
}

void Simulator::run() {
    cout << "Running Simulation... " << endl;

    unsigned int num_layers = circuit.cell_schedule.size();
    cout << "Total " << num_layers << " layers" << endl;

    ProgressBar progress_bar(num_layers);
    for (unsigned int i_layer = 0; i_layer < num_layers; i_layer++) {
        const auto& schedule_layer = circuit.cell_schedule[i_layer];
        for (auto* cell : schedule_layer) {
            Cell::build_bucket_index_schedule(cell->input_wires, INITIAL_CAPACITY);
        }

        int num_cells = schedule_layer.size();
        int num_finished_cells = 0;
        while (num_finished_cells < num_cells) {
            int prev_num_finished_gates = num_finished_cells;
            for (int i = 0; i < N_GATE_PARALLEL; i++) {
                const auto& cell = schedule_layer[num_finished_cells];
                if (cell->prepare_resource(resource_buffer)) {
                    num_finished_cells++;
                    if (num_finished_cells >= num_cells) break;
                }
            }
            const auto& batch_data = BatchResource{resource_buffer};
            resource_buffer.clear();
            simulate_batch<<<N_GATE_PARALLEL, N_STIMULI_PARALLEL>>>(batch_data);
            cudaDeviceSynchronize();
            for (int cell_idx = prev_num_finished_gates; cell_idx < num_finished_cells; cell_idx++) {
                const auto& cell = schedule_layer[cell_idx];
                cell->finalize();
            }
        }
        progress_bar.Progressed(i_layer);
    }
    cout << endl;
}
