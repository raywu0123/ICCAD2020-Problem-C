#ifndef ICCAD2020_DATA_STRUCTURES_H
#define ICCAD2020_DATA_STRUCTURES_H


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <utility>
#include <vector>
#include <unordered_map>

#include "constants.h"


struct TokenInfo {
    std::string wire_name;
    BitWidth bitwidth;
    size_t bucket_index;
};

struct Transition {
    Timestamp timestamp;
    char value;
    Transition(Timestamp t, char v): timestamp(t), value(v) {};
};

struct Bucket {
    Wirekey wirekey;
    std::vector<Transition> transitions;
    Bucket(const std::string& wire_name, int bit_index): wirekey(Wirekey{wire_name, bit_index}) {};
    explicit Bucket(Wirekey  wirekey): wirekey(std::move(wirekey)) {};
};

#endif
