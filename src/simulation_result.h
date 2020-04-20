#ifndef ICCAD2020_SIMULATION_RESULT_H
#define ICCAD2020_SIMULATION_RESULT_H

#include <vector>
#include <fstream>

#include "input_waveforms.h"
#include "circuit_model/circuit.h"
#include "simulator/data_structures.h"
#include "accumulators.h"


class SimulationResult {
public:
    explicit SimulationResult(
        Circuit& circuit, std::vector<std::string>& scopes, std::pair<int, std::string>& timescale_pair
    ): circuit(circuit), scopes(scopes), timescale_pair(timescale_pair) {};

    virtual void write(char* path);

    Circuit& circuit;
    std::vector<std::string>& scopes;
    std::pair<int, std::string>& timescale_pair;
    std::ofstream f_out;
};

class VCDResult : public SimulationResult {
public:
    explicit VCDResult(
        Circuit& circuit, std::vector<std::string>& scopes, std::pair<int, std::string>& timescale_pair
    );
    void write(char* path) override;

private:
    std::vector<Transition*> merge_sort();
    std::vector<VCDAccumulator*> accumulators;
};

class SAIFResult : public SimulationResult {
public:
    explicit SAIFResult(
        Circuit& circuit, std::vector<std::string>& scopes, std::pair<int, std::string>& timescale_pair
    );
    void write(char* path) override;
    std::vector<SAIFAccumulator*> accumulators;
};


#endif
