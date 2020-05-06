#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include <vector>

#include "circuit_model/circuit.h"
#include "simulation_result.h"

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

class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

    Circuit& circuit;
    ResourceBuffer resource_buffer;
};

#endif
