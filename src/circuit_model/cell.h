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

    Transition* alloc(int session_index) {
        if (session_index != previous_session_index) {
            first_free_data_ptr_index = 0;
            previous_session_index = session_index;
        }
        unsigned int size = capacity * N_STIMULI_PARALLEL;

        if (first_free_data_ptr_index >= data_ptrs.size())
            data_ptrs.push_back(MemoryManager::alloc(size));

        if (first_free_data_ptr_index >= data_ptrs.size())
            throw std::runtime_error("Invalid access to data_ptrs");

        Transition* data_ptr = data_ptrs[first_free_data_ptr_index];
        cudaMemset(data_ptr, 0, sizeof(Transition) * size);

        first_free_data_ptr_index++;
        return data_ptr;
    }
    virtual Transition* load(int session_index) { return alloc(session_index); }
    void free() {
        for (auto* data_ptr : data_ptrs) MemoryManager::free(data_ptr);
        data_ptrs.clear();
    };
    virtual void handle_overflow() {
        capacity *= 2;
        first_free_data_ptr_index = 0;
        free();
    }
    void store_to_bucket() const { wire->store_to_bucket(data_ptrs, capacity); }

    // records capacity
    Wire* wire;
    unsigned int capacity = INITIAL_CAPACITY;
    std::vector<Transition*> data_ptrs;

    unsigned int first_free_data_ptr_index = 0;
    int previous_session_index = -1;
};

struct ScheduledWire : public IndexedWire {
    explicit ScheduledWire(Wire* wire): IndexedWire(wire) {};

    Transition* load(int session_index) override {
//        FIXME what if bucket is empty?
        auto* ptr = IndexedWire::alloc(session_index);
        if (session_index > checkpoint.first) checkpoint = std::make_pair(session_index, bucket_idx);

        for (unsigned int stimuli_index = 0; stimuli_index < N_STIMULI_PARALLEL; stimuli_index++) {
            auto index = bucket_index_schedule[bucket_idx];
            auto size = bucket_index_schedule[bucket_idx + 1] - index;

            if (index != 0) index--, size++;  // leave one for delay calculation

            wire->load_from_bucket(ptr, capacity, stimuli_index, index, size);
            bucket_idx++;
            if (finished()) break;
        }
        return ptr;
    };
    void handle_overflow() override {
        IndexedWire::handle_overflow();
        bucket_idx = checkpoint.second;
    }
    bool finished() const { return bucket_idx + 1 >= bucket_index_schedule.size(); }
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
    std::pair<int, unsigned int> checkpoint = {0, 0};
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

    void set_paths(const std::vector<SDFPath>& ps);

    static void build_bucket_index_schedule(std::vector<ScheduledWire*>&, unsigned int);
    static unsigned int find_end_index(const Bucket&, unsigned int, Timestamp, unsigned int);

    bool finished() const;
    bool overflow() const;

    void init();
    void prepare_resource(int, ResourceBuffer&);
    void handle_overflow();
    void dump_result();

    std::vector<ScheduledWire*> input_wires;
    bool* overflow_ptr;

private:
    void build_wire_map(
        const StdCellDeclare* declare, const std::vector<PinSpec>& pin_specs,
        Wire* supply1_wire, Wire* supply0_wire
    );
    void create_wire_schedule(const std::vector<SubmoduleSpec>* submodule_specs);

    const ModuleSpec* module_spec;
    SDFSpec* sdf_spec = nullptr;

    std::vector<IndexedWire*> wire_schedule;
    std::unordered_map<unsigned int, IndexedWire*> wire_map;
    std::vector<IndexedWire*> cell_wires, output_wires;
};

#endif
