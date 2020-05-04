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
    InputInfo read_input_info();
    void read_input_waveforms(Circuit& circuit);
    void summary();

private:
    static void ignore_vcd_header(std::ifstream&);
    void read_vars();
    void get_buckets(Circuit& circuit);
    void read_dump();
    void read_single_time_dump(Timestamp);

    void emplace_transition(const std::string&, Timestamp, const std::string&);
    void emplace_transition(const std::string&, Timestamp, const char&);

    std::ifstream fin;

    std::unordered_map<std::string, TokenInfo> token_to_wire;
    unsigned int num_stimuli = 0;
    std::vector<Bucket*> buckets;

    int n_dump = 0;
};


#endif
