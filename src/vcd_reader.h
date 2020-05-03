#ifndef ICCAD2020_VCD_READER_H
#define ICCAD2020_VCD_READER_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "constants.h"
#include "circuit_model/circuit.h"
#include "simulator/data_structures.h"


class VCDReader {

public:
    explicit VCDReader(char* path) { fin = std::ifstream(path); };
    void read_input_waveforms(Circuit& circuit);
    void summary();

    InputInfo read_input_info();
    static void ignore_vcd_header(std::ifstream&);

    void read_vars_and_scopes();
    void get_buckets(Circuit& circuit);
    void read_dump();
    void read_single_time_dump(Timestamp);

    Bucket* emplace_transition(const std::string&, Timestamp, const std::string&);
    Bucket* emplace_transition(const std::string&, Timestamp, const char&);
    void update_stimuli_edge_indices(Bucket*);
    void finalize_stimuli_edge_indices();
    void push_back_stimuli_edge_indices();

    std::ifstream fin;

    std::unordered_map<std::string, TokenInfo> token_to_wire;
    unsigned int num_stimuli = 0;
    std::vector<Bucket*> buckets;
    unsigned num_buckets = 0;

    int n_dump = 0;
    size_t max_transition = 0;
    size_t min_transition = INT64_MAX;
    size_t sum_transition = 0;
};


#endif
