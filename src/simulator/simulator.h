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
    Simulator(
        Circuit& c,
        InputWaveforms& iw,
        SimulationResult* sr
    ): circuit(c), simulation_result(sr), input_waveforms(iw) {};

    void run();
    void simulate_batch_stimuli(unsigned int& i_batch);
    void set_input(unsigned int) const;

    BatchResource get_batch_data();

    Circuit& circuit;
    SimulationResult* simulation_result;
    InputWaveforms& input_waveforms;
    ResourceBuffer resource_buffer;
};

#endif
