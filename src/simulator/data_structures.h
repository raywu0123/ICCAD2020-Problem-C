#ifndef ICCAD2020_DATA_STRUCTURES_H
#define ICCAD2020_DATA_STRUCTURES_H


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <unordered_map>

#include "constants.h"


struct TokenInfo {
    string wire_name;
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
    vector<Transition> transitions;
    Bucket(const string& wire_name, int bit_index): wirekey(Wirekey{wire_name, bit_index}) {};
};


class WaveSet {
public:
    WaveSet();

    void summary() const;
    char* get_values_cuda();
    char* get_values_cpu() const;
    int size() const;

    char* values = nullptr;
    char* device_values = nullptr;

    size_t* timestamps = nullptr;
    size_t* device_timestamps = nullptr;

    int n_stimulus{};
    int stimuli_size{};
};

#endif //ICCAD2020_DATA_STRUCTURES_H
