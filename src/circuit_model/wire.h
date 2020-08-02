#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include <iostream>
#include <cassert>
#include <map>
#include "simulator/data_structures.h"
#include "simulator/memory_manager.h"
#include "simulator/collision_utils.h"

struct WireInfo {
    Wirekey wirekey;
    int bus_index;
};

using TransitionContainer = PinnedMemoryVector<Transition>;

struct Bucket {
    TransitionContainer transitions{ Transition{0, Values::Z} };

    void emplace_transition(Timestamp t, char r) {
        // for storing input
        auto& back = transitions.back();
        const auto& v = raw_to_enum(r);
        if (back.timestamp == 0 and t == 0 and v != back.value) back.value = v;
        else if (t > back.timestamp and v != back.value) transitions.emplace_back(t, v); // check validity of incoming transition
    }

    void reserve(unsigned int i) { transitions.reserve(i); }

    void push_back(const Data& data, bool verbose=false) {
        // for storing output
        unsigned int output_size;
        cudaMemcpy(&output_size, data.size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (verbose) std::cout << "output_size = " << output_size << std::endl;
        if (output_size == 0) return;

        Transition first_transition;
        cudaMemcpy(&first_transition, data.transitions, sizeof(Transition), cudaMemcpyDeviceToHost);
        if (verbose) std::cout << "first transition = " << first_transition << std::endl;

        const auto& t = first_transition.timestamp;
        const auto& v = first_transition.value;

        if (v == Values::PAD) return;  // batch contains no new transitions
        if (transitions.empty()) throw std::runtime_error("transitions is empty");

        Timestamp prev_t = transitions.back().timestamp;
        auto write_index = transitions.size();
        if (t <= prev_t) write_index = binary_search(transitions.data(), write_index - 1, t);
        auto offset = (write_index > 0 and v == transitions[write_index - 1].value) ? 1: 0;

        auto valid_data_size = output_size - offset;
        transitions.resize(write_index + valid_data_size);
        auto status =  cudaMemcpy(
            transitions.data() + write_index,
            data.transitions + offset,
            sizeof(Transition) * valid_data_size,
            cudaMemcpyDeviceToHost
        );
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorName(status));
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
    void set_drived();

    void load_from_bucket(
        Transition* ptr, unsigned int, unsigned int
    );
    void store_to_bucket(const std::vector<Data>& data_ptrs, unsigned int num_ptrs);

    std::vector<WireInfo> wire_infos;
    Bucket bucket;
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(Values value);
    Values value;
};

#endif
