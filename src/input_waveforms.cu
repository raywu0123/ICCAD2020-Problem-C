#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "input_waveforms.h"

extern double get_timescale(int, const string&);

void InputWaveforms::summary() const {
    cout << "Summary of Input Waveforms" << endl;
    cout << "Timescale: " << timescale << "s" << endl;
    cout << "Num dumps: " << n_dump << endl;
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
    do{fin >> s;}while(s != "$timescale");
}

void InputWaveforms::read_timescale() {
    string s, timescale_unit;
    int timescale_num;
    fin >> timescale_num >> timescale_unit >> s;
    timescale = get_timescale(timescale_num, timescale_unit);
}

void InputWaveforms::read_vars() {
    string s;
    do { getline(fin, s); } while (s.find("$var") != 0 or s.find("$var") == string::npos);
    do {
        string token, id;
        int n_bits;
        auto ss = istringstream(s);
        ss >> s >> s >> n_bits >> token >> id;

        char c;
        BitWidth bitwidth = {0, 0};
        if (n_bits > 1) ss >> c >> bitwidth.first >> c >> bitwidth.second;
        token_to_wire.emplace(token, make_pair(id, bitwidth));
        getline(fin, s);
    } while(s.find("$var") == 0);

    while (s.find("$dumpvars") == string::npos) { getline(fin, s); }
}


void InputWaveforms::build_buckets() {
    for (const auto& it : token_to_wire) {
        buckets.emplace(it.first, vector<pair<Timestamp, string>>{});
    }
}


void InputWaveforms::read_dump() {
    string s;
    getline(fin, s);
    Timestamp timestamp = time_tag_to_time(s);
    do {
        timestamp = read_single_time_dump(timestamp);
        n_dump++;
    } while (timestamp != -1);
}

Timestamp InputWaveforms::read_single_time_dump(Timestamp timestamp) {
    string s;
    getline(fin, s);
    while (s[0] != '#' and not fin.eof()) {
        string value, token;
        if (s[0] == 'b') {
            auto i = s.find(' ');
            value = s.substr(1, i - 1);
            token = s.substr(i + 1);
        } else {
            value = s.substr(0, 1);
            token = s.substr(1);
        }
        buckets[token].emplace_back(timestamp, value);
        getline(fin, s);
    }
    if (fin.eof()) return -1;
    return time_tag_to_time(s);
}


long long int InputWaveforms::time_tag_to_time(string& s) {
    s.erase(s.begin());
    return stoll(s);
}
