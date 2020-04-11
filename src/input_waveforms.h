#ifndef ICCAD2020_INPUT_WAVEFORMS_H
#define ICCAD2020_INPUT_WAVEFORMS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "constants.h"
#include "circuit_model/circuit.h"

using namespace std;


class InputWaveforms {

public:
    explicit InputWaveforms(char* path) { read(path); }
    void read(char*);
    void summary() const;

    void ignore_header();
    void read_timescale();
    void read_vars();
    void read_dump();
    Timestamp read_single_time_dump(Timestamp);
    void build_buckets();

    static long long int time_tag_to_time(string& s);
    ifstream fin;

    double timescale{};
    unordered_map<string, pair<string, BitWidth>> token_to_wire;
    unordered_map<string, vector<pair<Timestamp, string>>> buckets;
    int n_dump = 0;
};


#endif //ICCAD2020_INPUT_WAVEFORMS_H
