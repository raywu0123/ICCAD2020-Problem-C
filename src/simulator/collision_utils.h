#ifndef ICCAD2020_COLLISION_UTILS_H
#define ICCAD2020_COLLISION_UTILS_H

#include "data_structures.h"
#include "constants.h"

__host__ __device__ unsigned int binary_search(Transition* waveform, unsigned int end_index, Timestamp t);

__host__ __device__ void resolve_collisions_for_single_waveform(
    Transition* waveform, // (capacity)
    unsigned int capacity,
    unsigned int* length  // place to return length of resulting waveform
);

__host__ __device__ void resolve_collisions_for_batch_waveform(
    Transition* waveform, // (N_STIMULI_PARALLEL, capacity)
    unsigned int capacity,
    const unsigned int* stimuli_lengths, // (N_STIMULI_PARALLEL,)
    unsigned int* length,  // reference to lengths in Data structs
    unsigned int num_stimuli=N_STIMULI_PARALLEL
);

#endif //ICCAD2020_COLLISION_UTILS_H
