#ifndef ICCAD2020_VCD_READER_H
#define ICCAD2020_VCD_READER_H

#include <fstream>

#include "../graph.h"
#include "../input_waveforms.h"

using namespace std;


class VCDReader {
public:
    explicit VCDReader(InputWaveforms& iw, Graph& g):
        input_waveforms(iw), g(g) {};

    void read(char*);
    void summary();

    void ignore_header();
    void read_timescale();
    void read_vars();
    void read_dump();
    long long int read_single_time_dump(vector<pair<string, string>>&);

    static long long int time_tag_to_time(string& s);
    ifstream fin;

    InputWaveforms& input_waveforms;
    Graph& g;
};

#endif //ICCAD2020_VCD_READER_H
