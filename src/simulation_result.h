#ifndef ICCAD2020_SIMULATION_RESULT_H
#define ICCAD2020_SIMULATION_RESULT_H

#include "input_waveforms.h"
#include "circuit_model/circuit.h"
#include "simulator/data_structures.h"
#include "accumulators.h"


class SimulationResult {
public:
    explicit SimulationResult(Circuit& circuit): circuit(circuit) {};

    virtual void write(char* path) = 0;

    Circuit& circuit;
    std::vector<Accumulator*> accumulators;
};

class VCDResult : public SimulationResult {
public:
    explicit VCDResult(Circuit& circuit);
    void write(char* path) override;
};

class SAIFResult : public SimulationResult {
public:
    explicit SAIFResult(Circuit& circuit);
    void write(char* path) override;
};


#endif
