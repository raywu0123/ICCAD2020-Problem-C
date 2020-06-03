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
    Wire* wire;
    Transition* ptr = nullptr;
    std::vector<unsigned int> wire_schedule_indices;
    explicit IndexedWire(Wire* w) : wire(w) {};

    void init_wire_schedule_indices(const std::vector<Wire*>& wire_schedule) {
        for (unsigned int schedule_index = 0; schedule_index < wire_schedule.size(); schedule_index++) {
            if (wire_schedule[schedule_index] == wire) wire_schedule_indices.push_back(schedule_index);
        }
    };

    Transition* alloc() {
        if (ptr != nullptr) throw std::runtime_error("Illegal alloc of indexed wire");
        ptr = wire->alloc();
        return ptr;
    }
    void free() { wire->free(ptr); ptr = nullptr; };
    Transition* increase_capacity() {
        ptr = wire->increase_capacity();
        return ptr;
    }
};

struct ScheduledWire : public IndexedWire {
    explicit ScheduledWire(Wire* wire): IndexedWire(wire) {};

    void load_from_bucket() {
//        FIXME what if bucket is empty?
        unsigned int batch_bucket_idx = bucket_idx;
        for (unsigned int stimuli_index = 0; stimuli_index < N_STIMULI_PARALLEL; stimuli_index++) {
            auto index = bucket_index_schedule[batch_bucket_idx];
            auto size = bucket_index_schedule[batch_bucket_idx + 1] - index;

            if (index != 0) index--, size++;  // leave one for delay calculation

            wire->load_from_bucket(ptr, stimuli_index, index, size);
            batch_bucket_idx++;
            if (batch_bucket_idx + 1 >= bucket_index_schedule.size()) break;
        }
    };
    bool next() {
        bucket_idx += N_STIMULI_PARALLEL;
        return bucket_idx + 1 >= bucket_index_schedule.size();
    }

    void push_back_schedule_index(unsigned int i) {
        if (i > wire->bucket.size())
            throw std::runtime_error("Schedule index out of range.");
        if (not bucket_index_schedule.empty() and i < bucket_index_schedule.back())
            throw std::runtime_error("Schedule index in incorrect order.");

        bucket_index_schedule.push_back(i);
    }
    unsigned int size() const { return wire->bucket.size(); }
    std::vector<unsigned int> bucket_index_schedule{0};
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

    static void build_bucket_index_schedule(std::vector<ScheduledWire>&, unsigned int);
    static unsigned int find_end_index(const Bucket&, unsigned int, Timestamp, unsigned int);

    void init();
    void update_data_ptrs(Transition*, const IndexedWire&);
    void prepare_resource(ResourceBuffer&);
    void increase_capacity();
    bool overflow() const;
    bool next();
    void dump_result();

    void set_paths(const std::vector<SDFPath>& ps);

    std::vector<ScheduledWire> input_wires;
    std::vector<Transition*> data_ptrs;
    bool* overflow_ptr;

private:
    void build_wire_map(
        const StdCellDeclare* declare, const std::vector<PinSpec>& pin_specs,
        Wire* supply1_wire, Wire* supply0_wire
    );
    void add_cell_wire(Wire* wire_ptr);
    void create_wire_schedule(
        const std::vector<SubmoduleSpec>* submodule_specs
    );

    void init_wire_vectors(const StdCellDeclare* declare);

    const ModuleSpec* module_spec;
    SDFSpec* sdf_spec = nullptr;

    std::vector<Wire*> wire_schedule;
    std::unordered_map<unsigned int, Wire*> wire_map;

    std::vector<IndexedWire> cell_wires, output_wires;
};

#endif
