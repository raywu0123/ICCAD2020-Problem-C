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

using TransitionContainer = std::vector<Transition>;

struct Bucket {
    TransitionContainer transitions{ Transition{0, Values::Z} };

    void emplace_transition(const Timestamp& t, char r) {
        // for storing input
        auto& back = transitions.back();
        const auto& v = raw_to_enum(r);
        if (back.timestamp == 0 and t == 0 and v != back.value) back.value = v;
        else if (t > back.timestamp and v != back.value) transitions.emplace_back(t, v); // check validity of incoming transition
    }

    void reserve(unsigned int i) { transitions.reserve(i); }

    void push_back(const Transition* data, unsigned int size) {
        if (size == 0) return;
        const auto& t = data[0].timestamp;
        const auto& v = data[0].value;

        if (v == Values::PAD) return;  // batch contains no new transitions
        if (transitions.empty()) throw std::runtime_error("transitions is empty");

        Timestamp prev_t = transitions.back().timestamp;
        auto write_index = transitions.size();
        if (t <= prev_t) write_index = binary_search(transitions.data(), write_index - 1, t);
        auto offset = (write_index > 0 and v == transitions[write_index - 1].value) ? 1: 0;

        auto valid_data_size = size - offset;
        transitions.resize(write_index + valid_data_size);
        memcpy(transitions.data() + write_index, data + offset, sizeof(Transition) * valid_data_size);
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

    void to_device(cudaStream_t);
    void free_device();

    virtual void store_to_bucket(const std::vector<Data>&);
    virtual void emplace_transition(const Timestamp& t, char r);

    std::vector<WireInfo> wire_infos;
    Bucket bucket;
    bool is_constant = false;

    Transition *device_ptr = nullptr, *pinned_host_ptr = nullptr;
    int ref_count = 0;
};


class ConstantWire : public Wire {
    static bool store_to_bucket_warning_flag;
    static bool emplace_transition_warning_flag;

public:
    explicit ConstantWire(Values value): value(value) {
        is_constant = true;
        bucket.transitions.clear();
        bucket.transitions.emplace_back(0, value);
    }
    void store_to_bucket(const std::vector<Data>&) override {
        if (not store_to_bucket_warning_flag) {
            std::cerr << "| Warning: storing to constant wire\n";
            store_to_bucket_warning_flag = true;
        }
    };
    void emplace_transition(const Timestamp& t, char r) override {
        if (not emplace_transition_warning_flag) {
            std::cerr << "| Warning: emplacing transition to constant wire\n";
            emplace_transition_warning_flag = true;
        }
    }
    Values value;
};

#endif
