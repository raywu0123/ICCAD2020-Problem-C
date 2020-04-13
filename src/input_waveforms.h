#ifndef ICCAD2020_INPUT_WAVEFORMS_H
#define ICCAD2020_INPUT_WAVEFORMS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "constants.h"
#include "circuit_model/circuit.h"
#include "simulator/data_structures.h"


class InputWaveforms {

public:
    explicit InputWaveforms(char* path) { read(path); }
    void read(char*);
    void summary();

    void ignore_header();
    void read_timescale();
    void read_vars();
    void read_dump();
    void read_single_time_dump(Timestamp);

    void emplace_transition(const string&, Timestamp, const string&);
    void emplace_transition(const string&, Timestamp, const char&);
    void build_buckets();

    ifstream fin;

    double timescale{};
    unordered_map<string, TokenInfo> token_to_wire;
    vector<Bucket> buckets;
    unsigned num_buckets = 0;

    int n_dump = 0;
    int max_transition_index{};
    size_t max_transition = 0;
    size_t min_transition = INT64_MAX;
    size_t sum_transition = 0;
};


#endif
