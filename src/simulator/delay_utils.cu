#include "simulator.h"

extern __host__ __device__ int lookup_delay(
    unsigned int input_index, unsigned int output_index, char edge_type,
    const SDFSpec* sdf_spec
) {
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
    Transition** data, const unsigned int* capacities,
    const unsigned int* output_indices, unsigned int num_output, unsigned int num_input,
    const SDFSpec* sdf_spec
) {
    for (int i = 0; i < num_output; i++) {
        auto* output_data = data[output_indices[i]];
        auto capacity = capacities[output_indices[i]];
        for (unsigned int output_transition_index = 1; output_transition_index < capacity; output_transition_index++) {
            auto& output_transition = output_data[output_transition_index];
            if (output_transition.value == 0) break;

            output_transition.timestamp += lookup_delay(
                output_transition.delay_info.arg, i + num_input, output_transition.delay_info.edge_type,
                sdf_spec
            );
        }
    }
}
