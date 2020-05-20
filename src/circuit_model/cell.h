#ifndef ICCAD2020_CELL_H
#define ICCAD2020_CELL_H

#include <iostream>
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
        for (unsigned int stimuli_index = 0; stimuli_index < N_STIMULI_PARALLEL; stimuli_index++) {
            auto index = bucket_index_schedule[bucket_idx];
            auto size = bucket_index_schedule[bucket_idx + 1] - index;

            if (index != 0) index--, size++;  // leave one for delay calculation

            wire->load_from_bucket(stimuli_index, index, size);
            bucket_idx++;
            if (bucket_idx + 1 >= bucket_index_schedule.size()) break;
        }
        return bucket_idx + 1 >= bucket_index_schedule.size();
    };
    void push_back_schedule_index(unsigned int i) {
        if (i > wire->bucket.size())
            throw std::runtime_error("Schedule index out of range.");
        if (not bucket_index_schedule.empty() and i < bucket_index_schedule.back())
            throw std::runtime_error("Schedule index in incorrect order.");

        bucket_index_schedule.push_back(i);
    }
    unsigned int size() const { return wire->bucket.size(); }
    std::vector<unsigned int> bucket_index_schedule{0};
    Wire* wire;
    unsigned int bucket_idx = 0;
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

    static void build_bucket_index_schedule(std::vector<IndexedWire>&, unsigned int);
    static unsigned int find_end_index(const Bucket&, unsigned int, Timestamp, unsigned int);

    bool prepare_resource(ResourceBuffer&);
    void dump_result();

    void set_paths(const std::vector<SDFPath>& ps);

    std::vector<IndexedWire> input_wires;
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
};

#endif
