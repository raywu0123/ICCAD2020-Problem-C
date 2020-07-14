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
    explicit IndexedWire(Wire* w) : wire(w) {};

    Transition* alloc(int session_index);
    Transition* load(int session_index);
    virtual void free();
    void store_to_bucket() const;

    // records capacity
    Wire* wire;
    unsigned int capacity = INITIAL_CAPACITY;
    std::vector<Transition*> data_ptrs;

    unsigned int first_free_data_ptr_index = 0;
    int previous_session_index = -1;
};



struct ScheduledWire : public IndexedWire {
    explicit ScheduledWire(Wire* wire): IndexedWire(wire) {};

    unsigned int load(int session_index, const std::vector<unsigned int>&, unsigned int);
    void free() override;
    unsigned int size() const;
    std::vector<Transition> scheduled_bucket;
};


class Cell {
public:
    Cell(
        const ModuleSpec* module_spec,
        const std::vector<SubmoduleSpec>* submodule_specs,
        const StdCellDeclare* declare,
        const std::vector<PinSpec>&  pin_specs,
        Wire* supply1_wire, Wire* supply0_wire,
        std::string  name
    );
    ~Cell();

    void set_paths(const std::vector<SDFPath>& ps);

    bool finished() const;

    void init();
    static void build_scheduled_buckets(std::vector<ScheduledWire*>&, std::vector<unsigned int>&);
    void prepare_resource(int, ResourceBuffer&);
    void dump_result();

    std::vector<ScheduledWire*> input_wires;
    std::vector<unsigned int> starting_indices;
    std::string name;

private:
    void build_wire_map(
        const StdCellDeclare* declare, const std::vector<PinSpec>& pin_specs,
        Wire* supply1_wire, Wire* supply0_wire
    );
    void create_wire_schedule(const std::vector<SubmoduleSpec>* submodule_specs);

    const ModuleSpec* module_spec;
    SDFSpec* sdf_spec = nullptr;
    unsigned int capacity = INITIAL_CAPACITY;

    std::vector<IndexedWire*> wire_schedule;
    std::unordered_map<unsigned int, IndexedWire*> wire_map;
    std::vector<IndexedWire*> cell_wires, output_wires;

    unsigned int progress_index = 0;
};

#endif
