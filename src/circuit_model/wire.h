#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include <iostream>
#include <map>
#include "simulator/data_structures.h"
#include "simulator/memory_manager.h"
#include "simulator/collision_utils.h"
#include "accumulators.h"

struct WireInfo {
    Wirekey wirekey;
    int bus_index;
};

struct Bucket {
    std::vector<Transition> transitions;

    void push_back(const Transition* ptr, const unsigned int capacity) {
        Transition first_transition;
        cudaMemcpy(&first_transition, ptr, sizeof(Transition), cudaMemcpyDeviceToHost);
        const auto& t = first_transition.timestamp;
        const auto& v = first_transition.value;

        Timestamp prev_t = transitions.empty() ? LONG_LONG_MIN : transitions.back().timestamp;
        auto write_index = transitions.size();
        if (t <= prev_t) write_index = binary_search(transitions.data(), write_index - 1, t);
        auto offset = (write_index > 0 and v == transitions[write_index - 1].value) ? 1: 0;

        auto valid_data_size = capacity * N_STIMULI_PARALLEL - offset;
        transitions.resize(transitions.size() + valid_data_size);
        cudaMemcpy(
            transitions.data() + write_index,
            ptr,
            sizeof(Transition) * valid_data_size,
            cudaMemcpyDeviceToHost
        );

        // strip excess transitions
        for (unsigned int idx = write_index; idx < write_index + valid_data_size; idx++) {
            if (transitions[idx].value == 0) {
                transitions.resize(idx);
                break;
            }
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
    void free(Transition*);

    void load_from_bucket(Transition* ptr, unsigned int stimuli_index, unsigned int bucket_index, unsigned int size);
    void store_to_bucket();

    void reset_capacity();
    Transition* increase_capacity();

    std::vector<WireInfo> wire_infos;
    std::map<Transition*, unsigned int> data_ptrs;
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
