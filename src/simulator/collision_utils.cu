#include "simulator.h"

__host__ __device__ unsigned int binary_search(Transition* waveform, unsigned int end_index, Timestamp t) {
    if (waveform[0].timestamp >= t) return 0;

    unsigned int start_index = 0;
    while (start_index < end_index - 1) {
        auto mid_index = (start_index + end_index) / 2;
        if (waveform[mid_index].timestamp < t) {
            start_index = mid_index;
        } else {
            end_index = mid_index;
        }
    }
    return end_index;
}

extern __host__ __device__ void resolve_collisions_for_single_waveform(
    Transition* waveform, unsigned int capacity, unsigned int* length
) {
    unsigned int write_index = 0;
    Timestamp prev_t = LONG_LONG_MIN;
    for(unsigned int index = 0; index < capacity; index++) {
        if (waveform[index].value == 0) break;

        Timestamp& t = waveform[index].timestamp;
        if (t <= prev_t) write_index = binary_search(waveform, write_index - 1, t);
        waveform[write_index] = waveform[index];
        prev_t = t;
        write_index++;
    }
    *length = write_index;
}

extern __host__ __device__ void resolve_collisions_for_batch_waveform(
    Transition* waveform,
    unsigned int capacity,
    unsigned int* stimuli_lengths,
    unsigned int* length,
    unsigned int num_stimuli
) {
    unsigned int write_index = 0;
    Timestamp prev_t = LONG_LONG_MIN;
    for (unsigned int stimuli_index = 0; stimuli_index < num_stimuli; stimuli_index++) {
        Timestamp& t = waveform[capacity * stimuli_index].timestamp;
        unsigned int stimuli_length = stimuli_lengths[stimuli_index];
        if (t <= prev_t) write_index = binary_search(waveform, write_index - 1, t);
        for (unsigned int i = 0; i < stimuli_length; i++) {
            waveform[write_index + i] = waveform[capacity * stimuli_index + i];
        }
        write_index += stimuli_length;
        prev_t = waveform[capacity * stimuli_index + stimuli_length - 1].timestamp;
    }
    *length = write_index;
}