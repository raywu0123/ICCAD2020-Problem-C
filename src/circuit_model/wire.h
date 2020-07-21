#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include <iostream>
#include <map>
#include "simulator/data_structures.h"
#include "simulator/memory_manager.h"
#include "simulator/collision_utils.h"

struct WireInfo {
    Wirekey wirekey;
    int bus_index;
};

struct Bucket {
    std::vector<Transition> transitions{ Transition{0, 'x'} };

    void emplace_transition(Timestamp t, char v) {
        // for storing input
        auto& back = transitions.back();
        if (back.timestamp == 0 and t == 0 and v != back.value) back.value = v;
        else if (t > back.timestamp and v != back.value) transitions.emplace_back(t, v); // check validity of incoming transition
    }

    void push_back(const Transition* ptr, const unsigned int capacity, bool verbose=false) {
        // for storing output
        Transition first_transition;
        cudaMemcpy(&first_transition, ptr, sizeof(Transition), cudaMemcpyDeviceToHost);
        const auto& t = first_transition.timestamp;
        const auto& v = first_transition.value;

        if (v == 0) return;  // batch contains no new transitions
        if (transitions.empty()) throw std::runtime_error("transitions is empty");

        Timestamp prev_t = transitions.back().timestamp;
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
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorName(status));

        if (verbose) {
            for (int i = 0; i < valid_data_size; i++) std::cout << transitions[write_index + i];
            std::cout << "valid data size = " << valid_data_size << std::endl;
            std::cout << std::endl;
        }

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
    Transition& operator[] (size_t i) { return transitions[i]; }
    Transition operator[] (size_t i) const { return transitions[i]; }
    Transition& front() { return transitions.front(); }
    Transition& back() { return transitions.back(); }
};

class Wire {
public:
    Wire() = default;
    explicit Wire(const WireInfo&);

    void assign(const Wire&);

    static void load_from_bucket(
        Transition* ptr, const std::vector<Transition>&, unsigned int, unsigned int
    );
    void store_to_bucket(const std::vector<Transition*>& data_ptrs, unsigned int num_ptrs, unsigned int capacity);

    std::vector<WireInfo> wire_infos;
    Bucket bucket;
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value);
    char value;
};

#endif
