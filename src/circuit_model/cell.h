#ifndef ICCAD2020_CELL_H
#define ICCAD2020_CELL_H

#include <iostream>

#include "constants.h"
#include "wire.h"
#include "simulator/module_registry.h"
#include "simulator/containers.h"


struct InputWire {
    explicit InputWire(Wire* w, const CAPACITY_TYPE& capacity = INITIAL_CAPACITY) : wire(w), capacity(capacity) {};

    InputData& alloc(int session_index);
    InputData load(int session_index);
    unsigned int size() const;
    void handle_overflow();
    bool finished() const;
    void finish();

    void push_back_schedule_index(unsigned int i);

    Wire* wire;
    const CAPACITY_TYPE& capacity;
    std::vector<InputData> data_list;

    unsigned int first_free_data_ptr_index = 0;
    int previous_session_index = -1;

    std::vector<unsigned int> bucket_index_schedule{ 0 };
    unsigned int bucket_idx = 0;
    std::pair<int, unsigned int> checkpoint = {0, 0};
};


struct OutputWire {
    explicit OutputWire(Wire* w, const CAPACITY_TYPE& capacity = INITIAL_CAPACITY) : wire(w), capacity(capacity) {};

    Data load(OutputCollector<Transition>&, OutputCollector<unsigned int>&);
    void finish();
    void gather_result(Transition*, unsigned int*);
    void handle_overflow();

    // records capacity
    Wire* wire;
    const CAPACITY_TYPE& capacity;
    std::vector<Data> data_list;
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
    void init(ResourceCollector<SDFPath, Cell>&, ResourceCollector<Transition, Wire>&, OutputCollector<bool>&);
    void free();

    static void build_bucket_index_schedule(std::vector<InputWire*>& wires, unsigned int size);
    bool finished() const;
    void prepare_resource(
        int, ResourceBuffer&, bool* device_overflow,
        OutputCollector<Transition>&, OutputCollector<unsigned int>&,
        OutputCollector<Timestamp>&, OutputCollector<DelayInfo>&, OutputCollector<Values>&, OutputCollector<CAPACITY_TYPE>&
    );

    bool handle_overflow(bool*);

    void gather_results(Transition*, unsigned int*);

    std::vector<InputWire*> input_wires;
    std::vector<OutputWire*> output_wires;
    std::string name;
    std::vector<SDFPath> sdf_paths;
    const StdCellDeclare* declare;

private:
    void build_wire_map(const WireMap<Wire>& pin_specs);
    static unsigned int find_end_index(const Bucket&, unsigned int, const Timestamp&, unsigned int);

    const ModuleSpec* module_spec;
    NUM_ARG_TYPE num_args = 0;
    CAPACITY_TYPE output_capacity = INITIAL_CAPACITY;
    unsigned int overflow_offset = 0, sdf_offset = 0;
};

#endif
