#include "simulator.h"

__device__ __host__ unsigned int find_end_index(
    Transition* data, unsigned int size, unsigned int start_index, const Timestamp& t, const CAPACITY_TYPE& capacity
) {
    //    Binary Search for end_index <= t
    unsigned int low = start_index, high = min(start_index + capacity, size) - 1;
    if (data[high].timestamp <= t) return high;
    while (low < high) {
        unsigned mid = (low + high) / 2;
        if (mid == low) break;
        if (data[mid].timestamp < t) low = mid;
        else if (data[mid].timestamp > t) high = mid;
        else return mid;
    }
    return low;
}

__device__ __host__ void slice_waveforms(
    SliceInfo* s_slice_infos,
    Transition* const all_input_data, InputData* data, const CAPACITY_TYPE& capacity,
    const NUM_ARG_TYPE& num_wires,
    bool* overflow_ptr
) {
    Transition* input_data[MAX_NUM_MODULE_ARGS] = {nullptr};
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) input_data[i] = all_input_data + data[i].offset;
    unsigned int progress[MAX_NUM_MODULE_OUTPUT] = {0};

    NUM_ARG_TYPE num_finished = 0; bool finished[MAX_NUM_MODULE_ARGS] = {false};
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) if (capacity >= data[i].size) num_finished++;

    unsigned int write_stimuli_index = 1;
    while (num_finished < num_wires) {
        if (write_stimuli_index >= N_STIMULI_PARALLEL + 1) break;

        Timestamp min_end_timestamp = LONG_LONG_MAX;
        for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
            unsigned int end_index = progress[i] + capacity - 1;
            if (end_index >= data[i].size) continue;
            const auto& end_timestamp = input_data[i][end_index].timestamp;
            if (end_timestamp < min_end_timestamp) min_end_timestamp = end_timestamp;
        }

        for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
            if (write_stimuli_index > 0 and s_slice_infos[(write_stimuli_index - 1) * num_wires + i].offset == data[i].size) {
                s_slice_infos[write_stimuli_index * num_wires + i].offset = data[i].size;
            } else {
                auto start_index = input_data[i][progress[i]].timestamp > min_end_timestamp ?
                                   progress[i] - 1 : progress[i];
                auto end_index = find_end_index(input_data[i], data[i].size, start_index, min_end_timestamp, capacity);
                progress[i] = end_index + 1;
                s_slice_infos[write_stimuli_index * num_wires + i].offset = progress[i];
                if (progress[i] + capacity >= data[i].size and not finished[i]) {
                    finished[i] = true;
                    num_finished++;
                }
            }
        }
        write_stimuli_index++;
    }
    if (write_stimuli_index >= N_STIMULI_PARALLEL + 1) *overflow_ptr = true;
    else {
        for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
            s_slice_infos[write_stimuli_index * num_wires + i].offset = data[i].size;
        }
    }
}
