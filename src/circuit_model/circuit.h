#ifndef ICCAD2020_CIRCUIT_H
#define ICCAD2020_CIRCUIT_H

#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "constants.h"
#include "simulator/module_registry.h"
#include "input_waveforms.h"


using namespace std;

struct CellResource {

};

class Wire {

};

class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value): value(value) {};

    char value;
};

struct SDFPath {
    string in, out;
    int rising_delay, falling_delay;
};

class Cell {
public:
    Cell(
        const ModuleSpec* module_spec,
        const vector<SubmoduleSpec>* submodule_specs,
        const StdCellDeclare* declare,
        unordered_map<string, const Wire*>  io_wires,
        const Wire* supply1_wire, const Wire* supply0_wire
    ) : module_spec(module_spec), submodule_specs(submodule_specs), declare(declare),
        io_wires(std::move(io_wires)), supply1_wire(supply1_wire), supply0_wire(supply0_wire) {

    };

    CellResource prepare_resource() {
        return CellResource{};
    };

    void set_paths(const vector<SDFPath>& ps) { paths = ps; };

private:
    const ModuleSpec* module_spec;
    const vector<SubmoduleSpec>* submodule_specs;
    const StdCellDeclare* declare;
    unordered_map<string, const Wire*> io_wires;
    const Wire *supply1_wire, *supply0_wire;

    vector<SDFPath> paths;
};

struct PinSpec {
    string name;
    char type;
    Wirekey wirekey;
};


class Circuit {
public:
    explicit Circuit(const ModuleRegistry& module_registry): module_registry(module_registry) {
        register_01_wires();
    };

    void register_01_wires(); // register 1'b1 1'b0 wires
    const Wire* get_wire(const Wirekey&) const;
    void read_file(ifstream& fin, double input_timescale);
    void read_wires(ifstream& fin);
    void read_assigns(ifstream& fin);
    void read_cells(ifstream& fin);
    Cell* create_cell(const string&, const vector<PinSpec>&);

    void read_schedules(ifstream& fin);
    void read_sdf(ifstream& fin, double input_timescale);
    void bind_sdf_to_cell(const string&, const vector<SDFPath>&);

    void summary() const;

    void register_input_wires(const unordered_map<string, pair<string, BitWidth>>& token_to_wire);
    void register_input_wire(const Wirekey& wirekey);

    const ModuleRegistry& module_registry;

    string design_name;

    unordered_set<Wirekey, pair_hash> input_wires;
    unordered_map<Wirekey, const Wire*, pair_hash> wires;
    unordered_map<string, Cell*> cells;

    vector<vector<string>> cell_schedule;
    vector<vector<Wirekey>> wire_alloc_schedule;
    vector<vector<Wirekey>> wire_free_schedule;
};

#endif