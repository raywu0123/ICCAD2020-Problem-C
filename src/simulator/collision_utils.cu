#include <cassert>

#include "data_structures.h"
#include "constants.h"
#include "collision_utils.h"

__host__ __device__ unsigned int binary_search(const Transition* const waveform, unsigned int end_index, Timestamp t) {
    if (waveform[0].timestamp >= t) return 0;

    unsigned int start_index = 0;
    while (start_index < end_index) {
        auto mid_index = (start_index + end_index) / 2;
        if (mid_index == start_index) return end_index;

        if (waveform[mid_index].timestamp < t) start_index = mid_index;
        else end_index = mid_index;
    }
    return end_index;
}

extern __host__ __device__ void resolve_collisions_for_batch_waveform(
    Transition* waveform,
    const unsigned int* stimuli_lengths, unsigned int capacity,
    unsigned int num_stimuli
) {
    unsigned int write_index = 0;
    for (unsigned int stimuli_index = 0; stimuli_index < num_stimuli; stimuli_index++) {
        const unsigned int& stimuli_length = stimuli_lengths[stimuli_index];
        assert(stimuli_length <= capacity);
        if (stimuli_length == 0) continue;

        Timestamp& t = waveform[capacity * stimuli_index].timestamp;
        if (write_index >= 1 and t <= waveform[write_index - 1].timestamp){
            write_index = binary_search(waveform, write_index - 1, t);
        }

        auto offset = (write_index >= 1 and waveform[capacity * stimuli_index].value == waveform[write_index - 1].value) ? 1: 0;
        for (unsigned int i = offset; i < stimuli_length; i++) {
            assert(write_index < capacity * num_stimuli);
            waveform[write_index] = waveform[capacity * stimuli_index + i];
            write_index++;
        }
    }
    // add EOS token
    if (write_index < capacity * num_stimuli) waveform[write_index].timestamp = 0; waveform[write_index].value = 0;
}
