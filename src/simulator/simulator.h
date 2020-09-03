#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include <vector>

#include "circuit_model/circuit.h"
#include "simulation_result.h"

__host__ __device__ int lookup_delay(
    NUM_ARG_TYPE, NUM_ARG_TYPE, EdgeTypes, EdgeTypes,
    const SDFSpec*
);

__host__ __device__ void compute_delay(
    Transition**, const CAPACITY_TYPE& capacity, DelayInfo*,
    const NUM_ARG_TYPE&, const NUM_ARG_TYPE&,
    const SDFSpec* sdf_spec, CAPACITY_TYPE* lengths, bool verbose = false
);

__device__ __host__ void slice_waveforms(
    Timestamp* s_timestamps, DelayInfo* s_delay_infos, Values* s_values,
    Data* data, const CAPACITY_TYPE& capacity,
    const NUM_ARG_TYPE& num_wires, bool* overflow_ptr
);


class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

    Circuit& circuit;
};

#endif
