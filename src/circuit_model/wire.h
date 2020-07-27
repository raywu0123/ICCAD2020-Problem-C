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
    TransitionContainer transitions{ Transition{0, 'x'} };

    void emplace_transition(Timestamp t, char v) {
        // for storing input
        auto& back = transitions.back();
        if (back.timestamp == 0 and t == 0 and v != back.value) back.value = v;
        else if (t > back.timestamp and v != back.value) transitions.emplace_back(t, v); // check validity of incoming transition
    }

    void reserve(unsigned int i) { transitions.reserve(i); }

    void push_back(const Data& data, bool verbose=false) {
        // for storing output
        unsigned int output_size;
        cudaMemcpy(&output_size, data.size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (output_size == 0) return;

        Transition first_transition;
        cudaMemcpy(&first_transition, data.transitions, sizeof(Transition), cudaMemcpyDeviceToHost);
        const auto& t = first_transition.timestamp;
        const auto& v = first_transition.value;

        if (v == 0) return;  // batch contains no new transitions
        if (transitions.empty()) throw std::runtime_error("transitions is empty");

        Timestamp prev_t = transitions.back().timestamp;
        auto write_index = transitions.size();
        if (t <= prev_t) write_index = binary_search(transitions.data(), write_index - 1, t);
        auto offset = (write_index > 0 and v == transitions[write_index - 1].value) ? 1: 0;


        auto valid_data_size = output_size - offset;
        assert(valid_data_size <= INITIAL_CAPACITY * N_STIMULI_PARALLEL * 8);
        transitions.resize(write_index + valid_data_size);
        auto status =  cudaMemcpy(
            transitions.data() + write_index,
            data.transitions + offset,
            sizeof(Transition) * valid_data_size,
            cudaMemcpyDeviceToHost
        );
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorName(status));

        if (verbose) {
            std::cout << "first transition = " << first_transition << std::endl;
            std::cout << "v-back = " << transitions[write_index - 1].value << std::endl;
            for (int i = 0; i < valid_data_size; i++) std::cout << transitions[write_index + i];
            std::cout << "valid data size = " << valid_data_size << std::endl;
            std::cout << std::endl;
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
    Wire();
    ~Wire();
    explicit Wire(const WireInfo&);

    void assign(const Wire&);

    void load_from_bucket(
        Transition* ptr, unsigned int, unsigned int
    );
    void store_to_bucket(const std::vector<Data>& data_ptrs, unsigned int num_ptrs);

    std::vector<WireInfo> wire_infos;
    Bucket bucket;
    cudaStream_t stream;
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(char value);
    char value;
};

#endif
