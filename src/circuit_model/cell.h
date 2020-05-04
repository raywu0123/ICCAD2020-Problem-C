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

struct IndexedWire {
    explicit IndexedWire(Wire* wire): wire(wire) {};

    bool load_from_bucket() {
//        FIXME what if bucket is empty?
        wire->load_from_bucket(
            bucket_index_schedule[index],
            bucket_index_schedule[index + 1] - bucket_index_schedule[index]
        );
        index++;
        return (index >= bucket_index_schedule.size() - 1);
    };

    Wire* wire;
    unsigned int index = 0;
    std::vector<unsigned int> bucket_index_schedule;
};

class Cell {
public:
    Cell(
        const ModuleSpec* module_spec,
        const std::vector<SubmoduleSpec>* submodule_specs,
        const StdCellDeclare* declare,
        const std::vector<PinSpec>&  pin_specs,
        Wire* supply1_wire, Wire* supply0_wire,
        const std::vector<Wire*>& alloc_wires, const std::vector<Wire*>& free_wires
    );
    ~Cell();

    void build_bucket_index_schedule();
    bool prepare_resource(ResourceBuffer&);
    void finalize();

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
    SDFSpec* sdf_spec = nullptr;

    std::vector<Wire*> wire_schedule;
    std::unordered_map<unsigned int, Wire*> wire_map;

    std::vector<Wire*> cell_wires, output_wires;
    std::vector<IndexedWire> input_wires;
};

#endif
