#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "vcd_reader.h"


using namespace std;


void VCDReader::summary() {
    cout << "Summary of VCD File" << endl;
    cout << "Timescale: "
         << input_waveforms.timescale_num << " " << input_waveforms.timescale_unit << endl;
    cout << "Number of variables: " << input_waveforms.token_to_wire.size() << endl;
    cout << "Number of time dumps: " << input_waveforms.dumps.size() << endl;
    cout << "Begin time: " << input_waveforms.dumps.front().first << endl;
    cout << "End time: " << input_waveforms.dumps.back().first << endl;
    cout << endl;
}

void VCDReader::read(char* path) {
    fin = ifstream(path);
    ignore_header();
    read_timescale();
    read_vars();
    read_dump();
}

void VCDReader::ignore_header() {
    string s;
    do{fin >> s;}while(s != "$timescale");
}

void VCDReader::read_timescale() {
    string s;
    fin >> input_waveforms.timescale_num >> input_waveforms.timescale_unit >> s;
}

void VCDReader::read_vars() {
    string s;
    do { getline(fin, s); } while (s.find("$var") != 0 or s.find("$var") == string::npos);
    do {
        string variable, id;
        int n_bits;
        auto ss = istringstream(s);
        ss >> s >> s >> n_bits >> variable >> id;

        char c;
        pair<int, int> bitwidth;
        if (n_bits > 1) ss >> c >> bitwidth.first >> c >> bitwidth.second;
        input_waveforms.token_to_wire.emplace(variable, new Wire(id, bitwidth));
        getline(fin, s);
    } while(s.find("$var") == 0);

    while (s.find("$dumpvars") == string::npos) { getline(fin, s); }
}

void VCDReader::read_dump() {
    long long int time_tag = -1, next_time_tag = -1;
    string s;
    getline(fin, s);
    time_tag = time_tag_to_time(s);
    do {
        vector<pair<string, string>> single_time_dump;
        next_time_tag = read_single_time_dump(single_time_dump);
        input_waveforms.dumps.emplace_back(time_tag, single_time_dump);
        time_tag = next_time_tag;
    } while (time_tag != -1);
}

long long int VCDReader::read_single_time_dump(vector<pair<string, string>>& dump) {
    string s;
    do { getline(fin, s); } while (s[0] != '#' and not fin.eof());
    if (fin.eof()) return -1;
    return time_tag_to_time(s);
}


long long int VCDReader::time_tag_to_time(string& s) {
    s.erase(s.begin());
    return stoll(s);
}