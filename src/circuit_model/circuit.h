#ifndef ICCAD2020_CIRCUIT_H
#define ICCAD2020_CIRCUIT_H

#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cell.h"
#include "wire.h"

class Bus {
public:
    void init(const std::string&, const BitWidth&);

private:
    std::string name;
    BitWidth bitwidth;
    std::vector<char> state;
};

class BusManager {
public:
    void read(std::ifstream&);
    void add_transition(const std::vector<WireInfo>& wire_infos, const Transition& transition);

private:
    std::vector<Bus> buses;
};


class Circuit {
public:
    explicit Circuit(const ModuleRegistry& module_registry);
    ~Circuit();
    void summary() const;
    void read_file(std::ifstream& fin, double input_timescale, BusManager&, const std::string& output_flag);
    Wire* get_wire(const Wirekey&) const;

    std::string design_name;

    std::vector<std::vector<Cell*>> cell_schedule;
    std::vector<Wire*> wires;
    const ModuleRegistry& module_registry;

private:
    Wire* get_wire(unsigned int) const;
    void set_wire(unsigned int, Wire*);
    Cell* get_cell(const std::string& cell_id) const;

    void register_01_wires(const std::string& output_flag); // register 1'b1 1'b0 wires
    void read_wires(std::ifstream& fin, BusManager&, const std::string&);
    void read_assigns(std::ifstream& fin);
    void read_cells(std::ifstream& fin);
    Cell* create_cell(const std::string&, const std::vector<PinSpec>&, const std::vector<Wire*>&, const std::vector<Wire*>&);

    void read_schedules(std::ifstream& fin);
    void read_sdf(std::ifstream& fin, double input_timescale) const;
    void bind_sdf_to_cell(const std::string&, const std::vector<SDFPath>&) const;

    std::unordered_map<Wirekey, unsigned int, pair_hash> wirekey_to_index;
    std::unordered_map<std::string, Cell*> cells;
};

#endif