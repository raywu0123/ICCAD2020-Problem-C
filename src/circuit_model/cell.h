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

template<class T>
class WireMap {
public:
    T* get(unsigned int i) const {
        if (i >= MAX_NUM_MODULE_ARGS)
            throw std::runtime_error("Out-of-bounds access (" + std::to_string(i) + ") to getter of WireMap\n");
        auto* w = map[i];
        return w;
    }
    void set(unsigned int i, T* ptr) {
        if (i >= MAX_NUM_MODULE_ARGS)
            throw std::runtime_error("Out-of-bounds access (" + std::to_string(i) + ") to setter of WireMap\n");
        auto& entry = map[i];
        if (entry != nullptr) throw std::runtime_error("Duplicate setting to WireMap\n");
        entry = ptr;
    }

private:
    T* map[MAX_NUM_MODULE_ARGS] = { nullptr };
};

class Cell {
public:
    Cell(
        const ModuleSpec* module_spec,
        const StdCellDeclare* declare,
        const WireMap<Wire>&  pin_specs,
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
        const StdCellDeclare* declare, const WireMap<Wire>& pin_specs,
        Wire* supply1_wire, Wire* supply0_wire
    );

    const ModuleSpec* module_spec;
    SDFSpec* sdf_spec = nullptr;
    unsigned int num_args = 0;
    unsigned int capacity = INITIAL_CAPACITY;

    WireMap<IndexedWire> wire_map;
    std::vector<IndexedWire*> cell_wires, output_wires;

    unsigned int progress_index = 0;
};

#endif
