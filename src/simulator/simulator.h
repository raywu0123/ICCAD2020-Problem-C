#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include <vector>

#include "circuit_model/circuit.h"
#include "simulation_result.h"

__host__ __device__ int lookup_delay(
    unsigned int, unsigned int, char, char,
    const SDFSpec*
);

__host__ __device__ void compute_delay(
    Transition**, unsigned int,
    const unsigned int*, unsigned int, unsigned int,
    const SDFSpec* sdf_spec, unsigned int* lengths, bool verbose = false
);

class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

    Circuit& circuit;
};

#endif
