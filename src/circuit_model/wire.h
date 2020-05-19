#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include <iostream>
#include "simulator/data_structures.h"
#include "simulator/memory_manager.h"
#include "accumulators.h"

struct WireInfo {
    Wirekey wirekey;
    int bus_index;
};

struct DataPtr {
    DataPtr(Transition* data_ptr, unsigned int capacity): ptr(data_ptr), capacity(capacity) {};
    Transition* ptr;
    unsigned int capacity;
};

struct Bucket {
    std::vector<Transition> transitions;

    void push_back(const DataPtr& data_ptr) {
        unsigned int previous_size = transitions.size();
        transitions.resize(transitions.size() + data_ptr.capacity * N_STIMULI_PARALLEL);
        cudaMemcpy(
            transitions.data() + previous_size,
            data_ptr.ptr,
            sizeof(Transition) * data_ptr.capacity * N_STIMULI_PARALLEL,
            cudaMemcpyDeviceToHost
        );

        // strip excess transitions
        for (unsigned int idx = previous_size; idx < previous_size + data_ptr.capacity * N_STIMULI_PARALLEL; idx++) {
            if (transitions[idx].value == 0) {
                transitions.resize(idx);
                break;
            }
        }
    }

    void finalize() {
        for (const auto& tr : transitions) {
            std::cout << tr.timestamp << " " << tr.value << std::endl;
        }
    }

    unsigned int size() const {
        return transitions.size();
    }
};

class Wire {
public:
    Wire() = default;
    explicit Wire(const WireInfo&);

    void assign(const Wire&);
    Transition* alloc();
    void load_from_bucket(unsigned int stimuli_index, unsigned int bucket_index, unsigned int size);
    void store_to_bucket();

    void free();

    std::vector<WireInfo> wire_infos;
    std::vector<DataPtr> data_ptrs;
    Bucket bucket;
    unsigned int capacity = INITIAL_CAPACITY;
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value);
    char value;
    const unsigned int capacity = 1;
};

#endif
