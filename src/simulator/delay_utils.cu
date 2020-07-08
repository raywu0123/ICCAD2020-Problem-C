#include "simulator.h"

extern __host__ __device__ int lookup_delay(
    unsigned int input_index, unsigned int output_index, char input_edge_type, char output_edge_type,
    const SDFSpec* sdf_spec
) {
    int delay = 0;
    for (int i_row = 0; i_row < sdf_spec->num_rows; i_row++) {
        if (sdf_spec->input_index[i_row] == input_index and sdf_spec->output_index[i_row] == output_index) {
            if (sdf_spec->edge_type[i_row] == 'x' or sdf_spec->edge_type[i_row] == input_edge_type) {
                if (output_edge_type == '+') delay += sdf_spec->rising_delay[i_row];
                else if (output_edge_type == '-') delay += sdf_spec->falling_delay[i_row];
            }
        }
    }
    return delay;
}

__host__ __device__ char get_edge_type(char v1, char v2) {
    if (v2 == '1' or v1 == '0') return '+';
    if (v2 == '0' or v1 == '1') return '-';
    return 'x';
}

extern __host__ __device__ void compute_delay(
    Transition** data, const unsigned int* capacities,
    const unsigned int* output_indices, unsigned int num_output, unsigned int num_input,
    const SDFSpec* sdf_spec, unsigned int* lengths
) {
    for (int i = 0; i < num_output; i++) {
        auto* output_data = data[output_indices[i]];
        auto capacity = capacities[output_indices[i]];

        unsigned int write_idx = 1;
        unsigned int timeblock_start = 1, timeblock_end=1;
        while (timeblock_start < capacity and output_data[timeblock_start].value != 0) {
            const auto& t = output_data[timeblock_start].timestamp;
            const auto& v = output_data[timeblock_start].value;

            // find edges of timeblock
            while (output_data[timeblock_end].timestamp == t and output_data[timeblock_end].value != 0) timeblock_end++;

            // find index of previous transition
            auto prev_idx = write_idx - 1;
            if (t <= output_data[prev_idx].timestamp) prev_idx = binary_search(output_data, write_idx - 1, t) - 1;
            // find edge type
            const auto output_edge_type = get_edge_type(output_data[prev_idx].value, v);

            // find min_delay
            auto min_delay = LONG_LONG_MAX;
            for (unsigned int idx = timeblock_start; idx < timeblock_end; idx++) {
                auto d = lookup_delay(
                    output_data[idx].delay_info.arg, i + num_input,
                    output_data[idx].delay_info.edge_type, output_edge_type,
                    sdf_spec
                );
                min_delay = d < min_delay ? d : min_delay;
            }
            timeblock_start = timeblock_end; timeblock_end = timeblock_start;

            if (t + min_delay <= output_data[write_idx - 1].timestamp) write_idx = binary_search(output_data, write_idx - 1, t + min_delay);
            if (output_data[write_idx - 1].value == v) continue;
            output_data[write_idx].value = v; output_data[write_idx].timestamp = t + min_delay;
            write_idx++;
        }
        lengths[i] = write_idx;
    }
}
