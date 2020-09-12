#include <iostream>
#include <string>
#include <functional>
#include <cassert>
#include <chrono>

#include "vcd_reader.h"

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

void VCDReader::summary() const {
    cout << "Summary of Input Waveforms" << endl;
    cout << "Num dumps: " << n_dump << endl;
    cout << endl;
}

void VCDReader::read_input_waveforms(Circuit& circuit) {
    cout << "| STATUS: Reading Input VCD file...\n";
    auto t1 = chrono::high_resolution_clock::now();
    read_vars();
    get_buckets(circuit);
    read_dump();
    fin.close();
    auto t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << "| STATUS: VCDReader spent " << duration << " ms\n";
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
    for (auto& it : token_to_wire) {
        auto& token_info = it.second;
        token_info.bucket_index = wires.size();
        const auto& bitwidth = token_info.bitwidth;
        int step = bitwidth.first > bitwidth.second ? -1 : 1;
        bool is_vector = not (bitwidth.first == 0 and bitwidth.second == 0);
        for (int bit_index = bitwidth.first; bit_index != bitwidth.second + step; bit_index += step) {
            // buckets in MSB -> LSB order
            auto* wire = circuit.get_wire(Wirekey{token_info.wire_name, bit_index});
            wires.push_back(wire);
            if (wire->is_constant) {
                cerr << "| WARNING: Input VCD specified for constant wire " << token_info.wire_name;
                if (is_vector) cerr << "[" << bit_index << "]";
                cerr << "\n";
            }
        }
    }
}

void VCDReader::read_dump() {
    char c;
    fin >> c;

    bool is_dumpvars_section = (c != '#');
    if (is_dumpvars_section) fin.putback(c);

    while (not fin.eof()) {
        Timestamp t = 0;
        if (not is_dumpvars_section) fin >> t;
        read_single_time_dump(t);

        n_dump++;
        is_dumpvars_section = false;
    }
}

void VCDReader::read_single_time_dump(const Timestamp& timestamp) {
    char c;
    fin >> c;
    while (c != '#' and c != EOF and not fin.eof()) {
        string token;
        if (c == 'b') {
            string value;
            fin >> value >> token;
            emplace_transition(token, timestamp, value);
        } else if (c == '$') {
          // ignore $end of $dumpvars
          fin >> token;
        } else {
            fin >> token;
            emplace_transition(token, timestamp, c);
        }
        fin >> c;
    }
}

void VCDReader::emplace_transition(const string& token, const Timestamp& timestamp, const string& value) {
    const auto& it = token_to_wire.find(token);
    if (it == token_to_wire.end())
        throw runtime_error("Token " + token + " not found at t = " + to_string(timestamp) + "\n");
    const auto& token_info = it->second;

    const auto& bitwidth = token_info.bitwidth;
    unsigned int bit_range = abs(bitwidth.first - bitwidth.second) + 1;
    const auto& value_size = value.size();
    assert(bit_range >= value_size);
    unsigned pad_size = bit_range - value_size;
    for (unsigned int bit_index = 0; bit_index < pad_size; ++bit_index) {
        char bit_value = value[0] == '1' ? '0' : value[0];
        wires[token_info.bucket_index + bit_index]->emplace_transition(timestamp, bit_value);
    }
    for (unsigned int bit_index = pad_size; bit_index < bit_range; ++bit_index) {
        const char& bit_value = value[bit_index - pad_size];
        wires[token_info.bucket_index + bit_index]->emplace_transition(timestamp, bit_value);
    }
}

void VCDReader::emplace_transition(const string& token, const Timestamp& timestamp, const char& value) {
    const auto& it = token_to_wire.find(token);
    if (it == token_to_wire.end())
        throw runtime_error("Token " + token + " not found at t = " + to_string(timestamp) + "\n");
    const auto& token_info = it->second;
    auto* wire = wires[token_info.bucket_index];
    wire->emplace_transition(timestamp, value);
}
