#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "circuit_model/circuit.h"
#include "simulation_result.h"
#include "data_structures.h"


class Simulator {
public:
    Simulator(
        Circuit& c,
        InputWaveforms& iw,
        SimulationResult& sr
    ): circuit(c), simulation_result(sr), input_waveforms(iw) {};

    void run();
    void set_input(int);
    void simulate_batch(int);
    void simulate_batch_gates(int);

    Circuit& circuit;
    SimulationResult& simulation_result;
    InputWaveforms& input_waveforms;
};

#endif //ICCAD2020_SIMULATOR_H
