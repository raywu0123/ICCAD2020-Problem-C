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
    std::vector<Transition> transitions{ Transition{0, 'x'} };

    void emplace_transition(Timestamp t, char v) {
        // for storing input
        auto& back = transitions.back();
        if (t > back.timestamp and v != back.value) transitions.emplace_back(t, v); // check validity of incoming transition
    }

    void push_back(const Transition* ptr, const unsigned int capacity) {
        // for storing output
        Transition first_transition;
        cudaMemcpy(&first_transition, ptr, sizeof(Transition), cudaMemcpyDeviceToHost);
        const auto& t = first_transition.timestamp;
        const auto& v = first_transition.value;

        Timestamp prev_t = transitions.empty() ? LONG_LONG_MIN : transitions.back().timestamp;
        auto write_index = transitions.size();
        if (t <= prev_t) write_index = binary_search(transitions.data(), write_index - 1, t);
        auto offset = (write_index > 0 and v == transitions[write_index - 1].value) ? 1: 0;

        auto valid_data_size = capacity * N_STIMULI_PARALLEL - offset;
        transitions.resize(write_index + valid_data_size);
        auto status =  cudaMemcpy(
            transitions.data() + write_index,
            ptr + offset,
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

    void load_from_bucket(
        Transition* ptr, unsigned int capacity, unsigned int stimuli_index, unsigned int bucket_index, unsigned int size
    );
    void store_to_bucket(const std::vector<Transition*>& data_ptrs, unsigned int capacity);

    std::vector<WireInfo> wire_infos;
    Bucket bucket;
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value);
    char value;
};

#endif
