#ifndef ICCAD2020_SIMULATION_RESULT_H
#define ICCAD2020_SIMULATION_RESULT_H

#include <vector>
#include <fstream>

#include "vcd_reader.h"
#include "circuit_model/circuit.h"
#include "simulator/data_structures.h"


class SimulationResult {
public:
    explicit SimulationResult(
        const std::vector<Wire*>& wires, std::vector<std::string>& scopes, std::pair<int, std::string>& timescale_pair,
        Timestamp dumpon_time, Timestamp dumpoff_time,
        BusManager& bus_manager
    ):  wires(wires), scopes(scopes), timescale_pair(timescale_pair),
        dumpon_time(dumpon_time), dumpoff_time(dumpoff_time), bus_manager(bus_manager) {};

    virtual void write(char* path);

    const std::vector<Wire*>& wires;
    std::vector<std::string>& scopes;
    std::pair<int, std::string>& timescale_pair;
    Timestamp dumpon_time, dumpoff_time;
    BusManager& bus_manager;
    std::ofstream f_out;
};

struct PriorityQueueEntry {
    unsigned int bucket_index;
    Timestamp t;

    PriorityQueueEntry(unsigned int b_id, Timestamp t) : bucket_index(b_id), t(t) {};
    bool operator< (const PriorityQueueEntry& other) const { return t > other.t; } // smaller t, higher priority
};

class VCDResult : public SimulationResult {
public:
    explicit VCDResult(
        const std::vector<Wire*>& wires,
        std::vector<std::string>& scopes,
        std::pair<int, std::string>& timescale_pair,
        Timestamp dumpon_time, Timestamp dumpoff_time,
        BusManager& bus_manager
    );
    void write(char* path) override;
    static void group_timestamps(const std::vector<Timestamp>&, std::vector<std::pair<Timestamp, int>>&);

private:
    static void merge_sort(
        const std::vector<Wire*>&,
        std::vector<std::pair<unsigned int, unsigned int>>&,
        std::vector<Timestamp>&,
        Timestamp dumpon_time, Timestamp dumpoff_time
    );
    std::unordered_set<std::string> referenced_wire_ids;
};

class SAIFResult : public SimulationResult {
public:
    explicit SAIFResult(
        const std::vector<Wire*>& wires,
        std::vector<std::string>& scopes,
        std::pair<int, std::string>& timescale_pair,
        Timestamp dumpon_time, Timestamp dumpoff_time,
        BusManager& bus_manager
    );
    void write(char* path) override;
    static void calculate_wire_stats_vector(const std::vector<Wire*>& wires, Timestamp dumpon_time, Timestamp dumpoff_time);
    static void calculate_wire_stats(WireStat& wire_stat, const Bucket&, const Timestamp& dumpon_time, const Timestamp& dumpoff_time);
    void write_wirekey_result(const BitWidth& bitwidth, const Wirekey& wirekey, const WireStat& wirestat);

    std::string indent = "   ";
};


#endif
