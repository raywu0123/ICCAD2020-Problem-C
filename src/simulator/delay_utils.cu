#include <cassert>

#include "data_structures.h"
#include "simulator.h"

extern __host__ __device__ int lookup_delay(
    NUM_ARG_TYPE input_index, NUM_ARG_TYPE output_index, EdgeTypes input_edge_type, EdgeTypes output_edge_type,
    const SDFPath* sdf_paths, const unsigned int& sdf_num_rows
) {
    if (input_edge_type == EdgeTypes::NODELAY) return 0;

    int delay = 0;
    for (int i_row = 0; i_row < sdf_num_rows; i_row++) {
        if (sdf_paths[i_row].in == input_index
        and sdf_paths[i_row].out == output_index
        and (sdf_paths[i_row].edge_type == 'x' or sdf_paths[i_row].edge_type == edge_type_to_raw(input_edge_type))
        ) {
            if (output_edge_type == EdgeTypes::RISING) delay += sdf_paths[i_row].rising_delay;
            else if (output_edge_type == EdgeTypes::FALLING) delay += sdf_paths[i_row].falling_delay;
        }
    }
    assert(delay < 1000 and delay >= 0);
    return delay;
}

extern __host__ __device__ void compute_delay(
    Transition** data, const CAPACITY_TYPE& capacity, DelayInfo* delay_infos,
    const NUM_ARG_TYPE& num_output, const NUM_ARG_TYPE& num_input,
    const SDFPath* sdf_paths, const unsigned int& sdf_num_rows,
    CAPACITY_TYPE* lengths, bool verbose
) {
    for (NUM_ARG_TYPE i = 0; i < num_output; i++) {
        auto* output_data = data[i];
        if (output_data == nullptr) continue;

        CAPACITY_TYPE write_idx = 0;
        if (output_data[0].timestamp == -1) {
            output_data[0].timestamp = 0;
            write_idx = 1;
        }

        CAPACITY_TYPE timeblock_start = 1;
        Values prev_v = output_data[0].value;
        while (timeblock_start < capacity and output_data[timeblock_start].value != Values::PAD) {
            const auto& t = output_data[timeblock_start].timestamp;
            const auto& v = output_data[timeblock_start].value;
            // find edges of timeblock
            CAPACITY_TYPE num = 0;
            while ( timeblock_start < capacity
                and output_data[timeblock_start].timestamp == t
                and output_data[timeblock_start].value != Values::PAD
            ) {
                timeblock_start++; num++;
            }
            if (prev_v == v) continue;
            // find edge type
            const auto output_edge_type = get_edge_type(prev_v, v);
            prev_v = v;

            // find min_delay
            auto min_delay = LONG_LONG_MAX;
            for (CAPACITY_TYPE idx = 0; idx < num; idx++) {
                auto d = lookup_delay(
                    delay_infos[timeblock_start - num + idx].arg, i + num_input,
                    delay_infos[timeblock_start - num + idx].edge_type, output_edge_type,
                    sdf_paths, sdf_num_rows
                );
                min_delay = d < min_delay ? d : min_delay;
            }

            if (write_idx >= 1 and t + min_delay <= output_data[write_idx - 1].timestamp) {
                write_idx = binary_search(output_data, write_idx - 1, t + min_delay);
            }
            if (write_idx >= 1 and output_data[write_idx - 1].value == v) continue;

            assert(write_idx < capacity);
            output_data[write_idx].value = v; output_data[write_idx].timestamp = t + min_delay;
            write_idx++;
        }
        lengths[i] = write_idx;
    }
}
