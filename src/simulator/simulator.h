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
    Transition**, DelayInfo*,
    unsigned int, unsigned int,
    const SDFSpec* sdf_spec, unsigned int* lengths, bool verbose = false
);

__device__ __host__ void slice_waveforms(
    Timestamp s_timestamps[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    DelayInfo s_delay_infos[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    char s_values[N_STIMULI_PARALLEL][INITIAL_CAPACITY][MAX_NUM_MODULE_ARGS],
    Transition** data,
    unsigned int num_wires, unsigned int** progress_updates
);


class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

    Circuit& circuit;
};

#endif
