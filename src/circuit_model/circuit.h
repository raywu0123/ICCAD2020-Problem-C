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
#include "simulator/data_structures.h"
#include "accumulators.h"


class Wire {
public:
    void set_input(const vector<Transition>&, int start_index, int size);
    void alloc() {};
    void free() {
        accumulator->update();
    };
    Accumulator* accumulator = nullptr;
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
        const Wire* supply1_wire, const Wire* supply0_wire,
        vector<Wire*> alloc_wires, vector<Wire*> free_wires
    ) : module_spec(module_spec), submodule_specs(submodule_specs), declare(declare),
        io_wires(std::move(io_wires)), supply1_wire(supply1_wire), supply0_wire(supply0_wire),
        alloc_wires(std::move(alloc_wires)), free_wires(std::move(free_wires)),
        alloc_wires_size(alloc_wires.size()), free_wires_size(free_wires.size())
    {};

    CellResource prepare_resource() {
        for (unsigned i = 0; i < alloc_wires_size; i++) {
            const auto& wire_ptr = alloc_wires[i];
            wire_ptr->alloc();
        }
        return CellResource{};
    };

    void free_resource() {
        for (unsigned i = 0; i < free_wires_size; i++) {
            const auto& wire_ptr = free_wires[i];
            wire_ptr->free();
        }
    }

    void set_paths(const vector<SDFPath>& ps) { paths = ps; };

private:
    const ModuleSpec* module_spec;
    const vector<SubmoduleSpec>* submodule_specs;
    const StdCellDeclare* declare;

    unordered_map<string, const Wire*> io_wires;
    const Wire *supply1_wire, *supply0_wire;

    vector<Wire*> alloc_wires, free_wires;
    unsigned alloc_wires_size = 0, free_wires_size = 0;

    vector<SDFPath> paths;
};

struct PinSpec {
    string name;
    char type;
    Wire* wire;
};


class Circuit {
public:
    explicit Circuit(const ModuleRegistry& module_registry): module_registry(module_registry) {
        register_01_wires();
    };

    void register_01_wires(); // register 1'b1 1'b0 wires
    void register_input_wires(const vector<Bucket>&);

    Wire* get_wire(const Wirekey&) const;
    Wire* get_wire(unsigned int) const;
    void set_wire(unsigned int, Wire*);
    Cell* get_cell(const string& cell_id) const;

    void read_file(ifstream& fin, double input_timescale);
    void read_wires(ifstream& fin);
    void read_assigns(ifstream& fin);
    void read_cells(ifstream& fin);
    Cell* create_cell(const string&, const vector<PinSpec>&, const vector<Wire*>&, const vector<Wire*>&);

    void read_schedules(ifstream& fin);
    void read_sdf(ifstream& fin, double input_timescale) const;
    void bind_sdf_to_cell(const string&, const vector<SDFPath>&) const;

    void summary() const;

    const ModuleRegistry& module_registry;

    string design_name;

    vector<Wire*> wires;

    unordered_map<Wirekey, unsigned int, pair_hash> wirekey_to_index;
    vector<Wire*> input_wires;
    unordered_map<string, Cell*> cells;

    vector<vector<Cell*>> cell_schedule;
    vector<vector<Wire*>> wire_alloc_schedule;
    vector<vector<Wire*>> wire_free_schedule;
};

#endif