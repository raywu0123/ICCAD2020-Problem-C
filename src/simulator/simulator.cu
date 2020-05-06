#include "simulator/simulator.h"
#include "include/progress_bar.h"

using namespace std;

extern __host__ __device__ int lookup_delay(
    Transition* wire_data,
    unsigned int input_index,
    unsigned int output_index,
    unsigned int transition_index,
    const SDFSpec* sdf_spec
) {
    char edge_type;
    if (wire_data[transition_index].value == '1' or wire_data[transition_index - 1].value == '0') {
        edge_type = '+';
    } else if (wire_data[transition_index].value == '0' or wire_data[transition_index - 1].value == '1') {
        edge_type = '-';
    } else {
        edge_type = 'x';
    }
    int delay = 0;
    for (int i_row = 0; i_row < sdf_spec->num_rows; i_row++) {
        if (sdf_spec->input_index[i_row] == input_index and sdf_spec->output_index[i_row] == output_index) {
            if (sdf_spec->edge_type[i_row] == 'x' or sdf_spec->edge_type[i_row] == edge_type) {
//                TODO assuming rising_delay == falling_delay
                delay += sdf_spec->rising_delay[i_row];
            }
        }
    }
    return delay;
}

extern __host__ __device__ void compute_delay(
    Transition** data,
    unsigned int data_schedule_size,
    unsigned int* capacities,
    unsigned int* data_schedule_indices,
    unsigned int num_inputs, unsigned int num_outputs,
    const SDFSpec* sdf_spec
) {
    auto output_indices = new unsigned int[num_outputs];
    for (int i = 0; i < data_schedule_size; i++) {
        if (num_inputs <= data_schedule_indices[i] and data_schedule_indices[i] < num_inputs + num_outputs) {
            output_indices[data_schedule_indices[i] - num_inputs] = i;
        }
    }
    auto indices = new unsigned int[data_schedule_size];
    unsigned int num_finished = 0;
    for (int i = 0; i < data_schedule_size; i++) {
        if (   data_schedule_indices[i] >= num_inputs
            or data[i][1].value == 0
            or capacities[i] == 0) num_finished++;
        indices[i] = 0;
    }
    unsigned int output_transition_index = 1;

    while(num_finished < data_schedule_size) {
        Timestamp min_timestamp = LONG_LONG_MAX;
        unsigned int min_index;
        // find min timestamp
        for (int i = 0; i < data_schedule_size; i++) {
            if (data_schedule_indices[i] >= num_inputs) continue; // not an input wire
            if (indices[i] + 1 >= capacities[i]) continue;     // out of bound
            if (data[i][indices[i] + 1].value == 0) continue;  // is padding

            const auto& transition = data[i][indices[i] + 1];
            if (transition.timestamp < min_timestamp) {
                min_timestamp = transition.timestamp;
                min_index = i;
            }
        }
        indices[min_index]++;
        for (int output_index_ = 0; output_index_ < num_outputs; output_index_++) {
            const auto& output_data = data[output_indices[output_index_]];
            if (output_transition_index >= capacities[output_indices[output_index_]]) continue;
            output_data[output_transition_index].timestamp += lookup_delay(
                data[min_index],
                data_schedule_indices[min_index], num_inputs + output_index_,
                indices[min_index],
                sdf_spec
            );
        }
        if (   indices[min_index] >= capacities[min_index] - 1
            or data[min_index][indices[min_index] + 1].value == 0) {
            num_finished++;
        }
        output_transition_index++;
    }

    delete[] output_indices;
    delete[] indices;
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

__device__ void simulate_module(
    const ModuleSpec* module_spec,
    const SDFSpec* sdf_spec,
    Data* data_schedule
) {
    unsigned int data_schedule_idx = 0;
    for (int i = 0; i < module_spec->schedule_size; i++) {
        const auto& gate_fn_ptr = module_spec->gate_schedule[i];
        const auto& table = module_spec->tables[i];
        const auto& table_row_num = module_spec->table_row_num[i];
        const auto& num_inputs = module_spec->num_inputs[i];
        const auto& num_outputs = module_spec->num_outputs[i];
        simulate_gate_on_multiple_stimuli(
            gate_fn_ptr,
            data_schedule + data_schedule_idx,
            table,
            table_row_num,
            num_inputs, num_outputs
        );
        data_schedule_idx += num_inputs + num_outputs;
    }
    compute_delay_on_multiple_stimuli(data_schedule, module_spec, sdf_spec);
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
