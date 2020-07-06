#ifndef ICCAD2020_CIRCUIT_H
#define ICCAD2020_CIRCUIT_H

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cell.h"
#include "wire.h"
#include "utils.h"


class Bus {
public:
    void init(const std::string&, const BitWidth&);
    void update(const Transition& transition, int index);

    BitWidth bitwidth;
    std::string name;
    std::string state;
};

struct InputInfo {
    std::pair<int, std::string> timescale_pair;
    double timescale{};
    std::vector<std::string> scopes;

    explicit InputInfo(std::pair<int, std::string>  tp): timescale_pair(std::move(tp)) {
        timescale = get_timescale(timescale_pair.first, timescale_pair.second);
    }

    void summary() const {
        std::cout << "InputInfo Summary:" << std::endl;
        std::cout << "timescale_pair: " << timescale_pair.first << " " << timescale_pair.second << std::endl;
        std::cout << "timescale: " << timescale << std::endl;
        std::cout << "scopes: ";
        for (const auto& scope : scopes) std::cout << scope << " ";
        std::cout << std::endl << std::endl;
    }
};

class BusManager {
public:
    void read(std::ifstream&);
    std::string dumps_token_to_bus_map() const;
    void add_transition(const std::vector<WireInfo>&, const Transition&);
    std::string dumps_result();
    static std::string index_to_identifier(unsigned int);
    static std::string simplify_msb(const std::string&);

private:
    std::vector<Bus> buses;
    std::vector<std::string> index_to_identifier_map;
    std::unordered_set<std::pair<Bus*, unsigned int>, pair_hash> used_buses_in_current_time;
};

class Circuit {
public:
    explicit Circuit(const ModuleRegistry& module_registry);
    ~Circuit();
    void summary() const;

    void read_intermediate_file(std::ifstream& fin, double input_timescale, BusManager&);

    Wire* get_wire(const Wirekey&) const;

    std::string design_name;
    std::vector<std::vector<Cell*>> cell_schedule;
    std::vector<Wire*> wires;
    const ModuleRegistry& module_registry;

private:
    Wire* get_wire(unsigned int) const;
    void set_wire(unsigned int, Wire*);
    Cell* get_cell(const std::string& cell_id) const;

    void register_01_wires(); // register 1'b1 1'b0 wires
    void read_wires(std::ifstream& fin);
    void read_assigns(std::ifstream& fin);
    void read_cells(std::ifstream& fin);
    Cell* create_cell(const std::string&, const std::vector<PinSpec>&, const std::vector<Wire*>&, const std::vector<Wire*>&);

    void read_schedules(std::ifstream& fin);
    void read_sdf(std::ifstream& fin, double input_timescale) const;

    std::unordered_map<Wirekey, unsigned int, pair_hash> wirekey_to_index;
    std::unordered_map<std::string, Cell*> cells;
};

#endif