#include <iostream>
#include <string>
#include <functional>

#include "vcd_reader.h"
#include "utils.h"

using namespace std;


void VCDReader::ignore_vcd_header(ifstream& fin) {
    string s;
    do{ fin >> s; } while(s != "$timescale");
}

InputInfo VCDReader::read_input_info() {
    ignore_vcd_header(fin);
    string s;

    pair<int, string> timescale_pair;
    fin >> timescale_pair.first >> timescale_pair.second >> s;
    InputInfo info(timescale_pair);

    while (s != "$var") {
        fin >> s;
        if (s == "$scope") {
            fin >> s >> s;
            info.scopes.push_back(s);
        }
    }
    return info;
}

void VCDReader::summary() {
    cout << "Summary of Input Waveforms" << endl;
    cout << "Num dumps: " << n_dump << endl;
    cout << "Num stimuli: " << num_stimuli << endl;
    cout << endl;
}

void VCDReader::read_input_waveforms(Circuit& circuit) {
    cout << "Reading Input VCD file..." << endl;
    read_vars();
    get_buckets(circuit);
    read_dump();
}

void VCDReader::read_vars() {
    string s;
    do {
        string token, id;
        unsigned int n_bits;
        fin >> s >> n_bits>> token >> id;
        char c;
        BitWidth bitwidth = {0, 0};
        if (n_bits > 1) fin >> c >> bitwidth.first >> c >> bitwidth.second >> c;
        token_to_wire.emplace(token, TokenInfo{id, bitwidth, 0});
        fin >> s >> s;
    } while (s.find("$var") == 0);

    while (s.find("$dumpvars") == string::npos) { getline(fin, s); }
}

void VCDReader::get_buckets(Circuit& circuit) {
    for (auto& it : token_to_wire ) {
        auto& token_info = it.second;
        token_info.bucket_index = buckets.size();
        const auto& bitwidth = token_info.bitwidth;
        int bit_range = abs(bitwidth.first - bitwidth.second) + 1;
        for (int bit_index = 0; bit_index < bit_range; bit_index++) {
            const auto& wire = circuit.get_wire(Wirekey{token_info.wire_name, min(bitwidth.first, bitwidth.second) + bit_index});
            buckets.push_back(&wire->bucket);
        }
    }
}

void VCDReader::read_dump() {
    char c;
    fin >> c;
    while (not fin.eof()) {
        Timestamp t;
        fin >> t;
        read_single_time_dump(t);
        n_dump++;
    }
}

void VCDReader::read_single_time_dump(Timestamp timestamp) {
    char c;
    fin >> c;
    while (c != '#' and c != EOF and not fin.eof()) {
        string token;
        if (c == 'b') {
            string value;
            fin >> value >> token;
            emplace_transition(token, timestamp, value);
        } else {
            fin >> token;
            emplace_transition(token, timestamp, c);
        }
        fin >> c;
    }
}

void VCDReader::emplace_transition(const string& token, Timestamp timestamp, const string& value) {
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
        auto* bucket = buckets[token_info.bucket_index + bit_index];
        bucket->transitions.emplace_back(timestamp, bit_value);
    }
}

void VCDReader::emplace_transition(const string& token, Timestamp timestamp, const char& value) {
    const auto& it = token_to_wire.find(token);
    if (it == token_to_wire.end())
        throw runtime_error("Token " + token + " not found\n");
    const auto& token_info = it->second;
    auto* bucket = buckets[token_info.bucket_index];
    bucket->transitions.emplace_back(timestamp, value);
}
