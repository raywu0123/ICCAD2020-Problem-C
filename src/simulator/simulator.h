#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include <vector>

#include "circuit_model/circuit.h"
#include "simulation_result.h"

void __host__ __device__ get_output_indices(
    unsigned int* output_indices,
    unsigned int* data_schedule_indices, unsigned int data_schedule_size,
    unsigned int num_inputs, unsigned int num_outputs
);

__host__ __device__ int lookup_delay(
    Transition*,
    unsigned int, unsigned int,
    unsigned int,
    const SDFSpec*
);

__host__ __device__ void compute_delay(
    Transition** data_schedule,
    unsigned int data_schedule_size,
    unsigned int* capacities,
    unsigned int* data_schedule_indices,
    unsigned int num_inputs, unsigned int num_outputs,
    const SDFSpec* sdf_spec
);

__host__ __device__ void resolve_collisions_for_single_waveform(
    Transition* waveform, // (capacity)
    unsigned int capacity,
    unsigned int* length  // place to return length of resulting waveform
);

__host__ __device__ void resolve_collisions_for_batch_waveform(
    Transition* waveform, // (N_STIMULI_PARALLEL, capacity)
    unsigned int capacity,
    unsigned int* stimuli_lengths, // (N_STIMULI_PARALLEL,)
    unsigned int* length  // reference to lengths in Data structs
);

class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

    Circuit& circuit;
    ResourceBuffer resource_buffer;
};

#endif
