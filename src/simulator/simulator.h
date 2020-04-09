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
        const Circuit& c,
        InputWaveforms& iw,
        SimulationResult& sr
    ): simulation_result(sr){};

    void run();
    SimulationResult& simulation_result;
};

#endif //ICCAD2020_SIMULATOR_H
