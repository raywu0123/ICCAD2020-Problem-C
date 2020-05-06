#include "simulator.h"

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
    auto* output_indices = new unsigned int[num_outputs];
    get_output_indices(output_indices, data_schedule_indices, data_schedule_size, num_inputs, num_outputs);

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
