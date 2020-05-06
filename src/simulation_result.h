#ifndef ICCAD2020_SIMULATION_RESULT_H
#define ICCAD2020_SIMULATION_RESULT_H

#include <vector>
#include <fstream>

#include "vcd_reader.h"
#include "circuit_model/circuit.h"
#include "simulator/data_structures.h"
#include "circuit_model/accumulators.h"


class SimulationResult {
public:
    explicit SimulationResult(
        const std::vector<Wire*>& wires, std::vector<std::string>& scopes, std::pair<int, std::string>& timescale_pair
    ): wires(wires), scopes(scopes), timescale_pair(timescale_pair) {};

    virtual void write(char* path);

    const std::vector<Wire*>& wires;
    std::vector<std::string>& scopes;
    std::pair<int, std::string>& timescale_pair;
    std::ofstream f_out;
};

class VCDResult : public SimulationResult {
public:
    explicit VCDResult(
        const std::vector<Wire*>& wires,
        std::vector<std::string>& scopes,
        std::pair<int, std::string>& timescale_pair,
        BusManager& bus_manager
    );
    void write(char* path) override;
    static void group_timestamps(const std::vector<Timestamp>&, std::vector<std::pair<Timestamp, int>>&);

private:
    void merge_sort(std::vector<std::pair<unsigned int, unsigned int>>&, std::vector<Timestamp>&);
    BusManager& bus_manager;
};

class SAIFResult : public SimulationResult {
public:
    explicit SAIFResult(
        const std::vector<Wire*>& wires, std::vector<std::string>& scopes, std::pair<int, std::string>& timescale_pair
    );
    void write(char* path) override;
};


#endif
