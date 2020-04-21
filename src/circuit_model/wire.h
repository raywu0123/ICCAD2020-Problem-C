#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include "simulator/data_structures.h"
#include "simulator/memory_manager.h"
#include "accumulators.h"

struct WireInfo {
    Wirekey wirekey;
    int bus_index;
};

class Wire {
public:
    Wire() = default;
    Wire(const WireInfo&, const std::string&);
    explicit Wire(const std::string&);
    ~Wire();

    //    lifecycle: (set_input ->) alloc -> free
    void set_input(
        const std::vector<Transition>& ts,
        const std::vector<unsigned int>& stimuli_edges,
        unsigned int
    );
    void assign(const Wire&);
    void alloc();
    void free();

    std::vector<WireInfo> wire_infos;

    Transition* data_ptr = nullptr; // points to device memory
    unsigned int capacity = INITIAL_CAPACITY;
    Accumulator* accumulator = nullptr;
    Transition previous_transition{0, 'x'};
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value, const std::string& output_flag);
    char value;
    const unsigned int capacity = 1;
};

struct Bucket {
    Wirekey wirekey;
    Wire* wire_ptr;
    std::vector<Transition> transitions;
    std::vector<unsigned int> stimuli_edge_indices{0};
    Bucket(const std::string& wire_name, int bit_index): wirekey(Wirekey{wire_name, bit_index}) {};
    explicit Bucket(Wirekey  wirekey): wirekey(std::move(wirekey)) {};
};

#endif
