#ifndef ICCAD2020_CELL_H
#define ICCAD2020_CELL_H

#include "constants.h"
#include "wire.h"
#include "simulator/module_registry.h"

struct SDFPath {
    unsigned int in, out;
    char edge_type;
    int rising_delay, falling_delay;
};

struct PinSpec {
    unsigned int index{};
    Wire* wire{};
    PinSpec() = default;
    PinSpec(unsigned int index, Wire* wire): index(index), wire(wire) {};
};

class Cell {
public:
    Cell(
        const ModuleSpec* module_spec,
        const std::vector<SubmoduleSpec>* submodule_specs,
        const StdCellDeclare* declare,
        const std::vector<PinSpec>&  pin_specs,
        Wire* supply1_wire, Wire* supply0_wire,
        std::vector<Wire*> alloc_wires, std::vector<Wire*> free_wires
    );
    ~Cell();

    void prepare_resource(ResourceBuffer&);
    bool is_finished();
    void finalize_output();
    void free_resource();


    void set_paths(const std::vector<SDFPath>& ps);

private:
    void build_wire_map(
        const StdCellDeclare* declare, const std::vector<PinSpec>& pin_specs,
        Wire* supply1_wire, Wire* supply0_wire
    );
    void add_cell_wire(Wire* wire_ptr);
    void create_wire_schedule(
        const std::vector<SubmoduleSpec>* submodule_specs
    );

    const ModuleSpec* module_spec;
    std::vector<const Wire*> wire_schedule;
    std::unordered_map<unsigned int, Wire*> wire_map;

    std::vector<Wire*> cell_wires;
    std::vector<Wire*> alloc_wires, free_wires;
    unsigned alloc_wires_size, free_wires_size;

    SDFSpec* sdf_spec = nullptr;
};

#endif
