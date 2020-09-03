#ifndef ICCAD2020_CELL_H
#define ICCAD2020_CELL_H

#include <iostream>

#include "constants.h"
#include "wire.h"
#include "simulator/module_registry.h"
#include "simulator/containers.h"

struct SDFPath {
    NUM_ARG_TYPE in, out;
    char edge_type;
    int rising_delay, falling_delay;
};


struct IndexedWire {
    explicit IndexedWire(Wire* w, const CAPACITY_TYPE& capacity = INITIAL_CAPACITY) : wire(w), capacity(capacity) {};

    Data alloc(int session_index);
    virtual Data load(int session_index);
    virtual void free();
    virtual void finish();
    void store_to_bucket() const;

    virtual void handle_overflow();

    // records capacity
    Wire* wire;
    const CAPACITY_TYPE& capacity;
    std::vector<Data> data_list;

    unsigned int first_free_data_ptr_index = 0;
    int previous_session_index = -1;
};


struct ScheduledWire : public IndexedWire {
    explicit ScheduledWire(Wire* wire, const CAPACITY_TYPE& capacity = INITIAL_CAPACITY): IndexedWire(wire, capacity) {};

    Data load(int session_index) override;
    void free() override;
    unsigned int size() const;
    void handle_overflow() override;
    bool finished() const;
    void finish() override;

    void push_back_schedule_index(unsigned int i);

    std::vector<unsigned int> bucket_index_schedule{ 0 };
    unsigned int bucket_idx = 0;
    std::pair<int, unsigned int> checkpoint = {0, 0};
};

template<class T>
class WireMap {
public:
    T* get(NUM_ARG_TYPE i) const {
        if (i >= MAX_NUM_MODULE_ARGS)
            throw std::runtime_error("Out-of-bounds access (" + std::to_string(i) + ") to getter of WireMap\n");
        auto* w = map[i];
        return w;
    }
    void set(NUM_ARG_TYPE i, T* ptr) {
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
        std::string  name
    );
    void init();
    void free();

    static void build_bucket_index_schedule(std::vector<ScheduledWire*>& wires, unsigned int size);
    bool finished() const;
    void prepare_resource(int, ResourceBuffer&);
    bool gather_results();

    std::vector<ScheduledWire*> input_wires;
    std::vector<IndexedWire*> output_wires;
    std::string name;
    std::vector<SDFPath> sdf_paths;
    const StdCellDeclare* declare;

private:
    void build_wire_map(const WireMap<Wire>& pin_specs);
    void set_paths();
    bool handle_overflow();
    static unsigned int find_end_index(const Bucket&, unsigned int, const Timestamp&, unsigned int);

    const ModuleSpec* module_spec;
    SDFSpec* sdf_spec = nullptr;
    SDFSpec host_sdf_spec{};
    NUM_ARG_TYPE num_args = 0;
    CAPACITY_TYPE output_capacity = INITIAL_CAPACITY;
    bool* overflow_ptr = nullptr;

    WireMap<IndexedWire> wire_map;
};

#endif
