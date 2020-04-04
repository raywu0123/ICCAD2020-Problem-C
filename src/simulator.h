#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include "circuit_model/circuit.h"
#include "simulation_result.h"

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
