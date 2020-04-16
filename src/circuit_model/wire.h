#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include "simulator/data_structures.h"
#include "simulator/memory_manager.h"
#include "accumulators.h"


class Wire {
public:
//    lifecycle: (set_input ->) alloc -> free

    void set_input(
        const std::vector<Transition>& ts,
        const std::vector<unsigned int>& stimuli_edges,
        unsigned int
    );
    void alloc();
    void free();
    Transition* data_ptr = nullptr; // points to device memory
    unsigned int capacity = INITIAL_CAPACITY;
    Accumulator* accumulator = nullptr;
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value);
    char value;
    const unsigned int capacity = 1;
};

#endif
