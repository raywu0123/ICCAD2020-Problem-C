#ifndef ICCAD2020_WIRE_H
#define ICCAD2020_WIRE_H

#include <iostream>
#include <cassert>
#include <map>

#include "simulator/data_structures.h"
#include "simulator/collision_utils.h"
#include "utils.h"

struct WireInfo {
    Wirekey wirekey;
    int bus_index;
};

using TransitionContainer = PinnedMemoryVector<Transition>;

class Wire {
public:
    Wire() = default;
    explicit Wire(const WireInfo&);

    void assign(const Wire&);
    void set_drived();

    void emplace_transition(Timestamp t, char r) {
        // for storing input
        auto& back = bucket.back();
        const auto& v = raw_to_enum(r);
        if (back.timestamp == 0 and t == 0 and v != back.value) back.value = v;
        else if (t > back.timestamp and v != back.value) bucket.emplace_back(t, v); // check validity of incoming transition
    }

    std::vector<WireInfo> wire_infos;
    TransitionContainer bucket{Transition{0, Values::Z} };
};


class ConstantWire : public Wire {
public:
    explicit ConstantWire(Values value);
    Values value;
};

#endif
