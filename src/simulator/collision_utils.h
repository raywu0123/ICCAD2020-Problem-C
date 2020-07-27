#ifndef ICCAD2020_COLLISION_UTILS_H
#define ICCAD2020_COLLISION_UTILS_H

__host__ __device__ unsigned int binary_search(const Transition* waveform, unsigned int end_index, Timestamp t);

__host__ __device__ void resolve_collisions_for_batch_waveform(
    Transition* waveform, // (N_STIMULI_PARALLEL, capacity)
    const unsigned int* stimuli_lengths, unsigned int capacity, // (N_STIMULI_PARALLEL,)
    unsigned int num_stimuli=N_STIMULI_PARALLEL
);

#endif //ICCAD2020_COLLISION_UTILS_H
