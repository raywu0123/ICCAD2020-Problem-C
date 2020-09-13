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


template<class T>
class StringMap {

public:
    StringMap() = default;

    void reserve(size_t max_length) {
        assert(max_length > 0);
        size_t size = pow(126 - 32 + 1,  max_length);
        vec.resize(size);
    }

    size_t hash_function(const std::string& key) {
        size_t index = 0;
        for (const auto& c : key) index = index * (126 - 32 + 1) + (c - 32);
        return index;
    }

    T operator[] (const std::string& key) const {
        return vec[hash_function(key)];
    }
    T& operator[] (const std::string& key) {
        return vec[hash_function(key)];
    }
    std::vector<T> vec;
};

class VCDReader {

public:
    explicit VCDReader(char* path) { fin = std::ifstream(path); };
    InputInfo read_input_info();
    void read_input_waveforms(Circuit& circuit);
    void summary() const;

private:
    static void ignore_vcd_header(std::ifstream&);
    void read_vars();
    void get_buckets(Circuit& circuit);
    void read_dump();
    void read_single_time_dump(const Timestamp&);

    void emplace_transition(const std::string&, const Timestamp&, const std::string&);
    void emplace_transition(const std::string&, const Timestamp&, const char&);

    std::ifstream fin;

    StringMap<TokenInfo*> token_to_wire;
    std::vector<std::pair<std::string, TokenInfo>> token_and_infos;
    std::vector<Wire*> wires;

    unsigned int n_dump = 0;
};


#endif
