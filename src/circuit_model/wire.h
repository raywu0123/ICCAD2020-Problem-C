#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include "simulator/data_structures.h"
#include "simulator/memory_manager.h"
#include "accumulators.h"

struct WireInfo {
    Wirekey wirekey;
    int bus_index;
};

struct Bucket {
    std::vector<Transition> transitions;
    std::vector<unsigned int> stimuli_edge_indices{0};
};


class Wire {
public:
    Wire() = default;
    explicit Wire(const WireInfo&);

    void assign(const Wire&);

    std::vector<WireInfo> wire_infos;

    Transition* data_ptr = nullptr; // points to device memory
    unsigned int capacity = INITIAL_CAPACITY;
    Bucket bucket;
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value);
    char value;
    const unsigned int capacity = 1;
};

#endif
