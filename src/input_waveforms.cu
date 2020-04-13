#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include "input_waveforms.h"

extern double get_timescale(int, const string&);


void InputWaveforms::summary() {
    cout << "Summary of Input Waveforms" << endl;
    cout << "Timescale: " << timescale << "s" << endl;
    cout << "Num dumps: " << n_dump << endl;

    for (int i_bucket = 0; i_bucket < buckets.size(); i_bucket++) {
        const auto& bucket = buckets[i_bucket];
        sum_transition += bucket.transitions.size();
        if (bucket.transitions.size() > max_transition) {
            max_transition = bucket.transitions.size();
            max_transition_index = i_bucket;
        }
        if (bucket.transitions.size() < min_transition)
            min_transition = bucket.transitions.size();
    }
    cout << "Max transition: " << max_transition << endl;
    cout << "Mean transition: " << double(sum_transition) / double(buckets.size()) << endl;
    cout << "Min transition: " << min_transition << endl;
    cout << endl;
}

void InputWaveforms::read(char* path) {
    fin = ifstream(path);
    ignore_header();
    read_timescale();
    read_vars();
    build_buckets();
    read_dump();
}

void InputWaveforms::ignore_header() {
    string s;
    do{fin >> s;} while(s != "$timescale");
}

void InputWaveforms::read_timescale() {
    string s, timescale_unit;
    int timescale_num;
    fin >> timescale_num >> timescale_unit >> s;
    timescale = get_timescale(timescale_num, timescale_unit);
}

void InputWaveforms::read_vars() {
    string s;
    fin >> s;
    while (s != "$var") fin >> s;

    while (s.find("$var") == 0) {
        string token, id;
        unsigned int n_bits;
        fin >> s >> n_bits>> token >> id;
        char c;
        BitWidth bitwidth = {0, 0};
        if (n_bits > 1) fin >> c >> bitwidth.first >> c >> bitwidth.second >> c;
        token_to_wire.emplace(token, TokenInfo{id, bitwidth, 0});
        fin >> s >> s;
    }

    while (s.find("$dumpvars") == string::npos) { getline(fin, s); }
}


void InputWaveforms::build_buckets() {
    for (auto& it : token_to_wire) {
        auto& token_info = it.second;
        const BitWidth& bitwidth = token_info.bitwidth;

        token_info.bucket_index = buckets.size();
        int min_bit_index = min(bitwidth.first, bitwidth.second);
        int max_bit_index = max(bitwidth.first, bitwidth.second);
        for (int bit_index = min_bit_index; bit_index <= max_bit_index; bit_index++) {
            buckets.emplace_back(token_info.wire_name, bit_index);
        }
        num_buckets += max_bit_index - min_bit_index + 1;
    }
}


void InputWaveforms::read_dump() {
    char c;
    fin >> c;
    while (not fin.eof()) {
        Timestamp t;
        fin >> t;
        read_single_time_dump(t);
        n_dump++;
    }
}

void InputWaveforms::read_single_time_dump(Timestamp timestamp) {
    char c;
    fin >> c;
    while (c != '#' and c != EOF and not fin.eof()) {
        string value, token;
        if (c == 'b') {
            fin >> value >> token;
        } else {
            fin >> token;
            value = string{c};
        }
        emplace_transition(token, timestamp, value);
        fin >> c;
    }
}

void InputWaveforms::emplace_transition(const string& token, Timestamp timestamp, const string& value) {
    const auto& it = token_to_wire.find(token);
    if (it == token_to_wire.end())
        throw runtime_error("Token " + token + " not found\n");
    const auto& token_info = it->second;

    const auto& bitwidth = token_info.bitwidth;
    int bit_range = abs(bitwidth.first - bitwidth.second) + 1;
    int pad_size = bit_range - value.size();
    if (pad_size < 0) {
        throw runtime_error(
            "Value: " + value +
            " and bitwidth: " + to_string(bitwidth.first) + " " + to_string(bitwidth.second) + " incompatible"
        );
    }
    for (int bit_index = 0; bit_index < bit_range; bit_index++) {
        char bit_value;
        if (bit_index - pad_size < 0) {
            if (value[0] == '1') bit_value = '0';
            else bit_value = value[0];
        } else {
            bit_value = value[bit_index - pad_size];
        }
        auto& bucket = buckets[token_info.bucket_index + bit_index];
        bucket.transitions.emplace_back(timestamp, bit_value);
    }
}
