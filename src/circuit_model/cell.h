#ifndef ICCAD2020_CELL_H
#define ICCAD2020_CELL_H

#include <iostream>
#include <queue>

#include "constants.h"
#include "wire.h"
#include "simulator/module_registry.h"
#include "simulator/job.h"

struct SDFPath {
    unsigned int in, out;
    char edge_type;
    int rising_delay, falling_delay;
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
        std::string  name
    );
    static unsigned int build_bucket_index_schedule(std::vector<InputWire*>& wires, unsigned int size);

    void push_jobs(std::queue<Job*>&);
    void finish();

    std::vector<InputWire*> input_wires;
    std::vector<OutputWire*> output_wires;
    std::string name;
    std::vector<SDFPath> sdf_paths;

private:
    void init();
    void build_wire_map(
        const StdCellDeclare* declare, const WireMap<Wire>& pin_specs
    );
    void set_paths();
    static unsigned int find_end_index(const PinnedMemoryVector<Transition>&, unsigned int, const Timestamp&, unsigned int);

    const ModuleSpec* module_spec;
    SDFSpec* sdf_spec = nullptr;
    SDFSpec host_sdf_spec{};
    unsigned int num_args = 0;

    unsigned int schedule_size = 0;
    WireMap<WrappedWire> wire_map;
};

#endif
