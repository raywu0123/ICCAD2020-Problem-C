#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include <vector>

#include "circuit_model/circuit.h"
#include "simulation_result.h"

__host__ __device__ int lookup_delay(
    unsigned int, unsigned int, EdgeTypes, EdgeTypes,
    const SDFSpec*
);

__host__ __device__ void compute_delay(
    Transition**, unsigned int capacity, DelayInfo*,
    unsigned int, unsigned int,
    const SDFSpec* sdf_spec, unsigned int* lengths, bool verbose = false
);

__device__ __host__ void slice_waveforms(
    Timestamp* s_timestamps, DelayInfo* s_delay_infos, Values* s_values,
    Data* data, unsigned int capacity,
    unsigned int num_wires, bool* overflow_ptr
);


class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

    Circuit& circuit;
};

#endif
