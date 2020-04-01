#ifndef ICCAD2020_INTERMEDIATE_FILE_READER_H
#define ICCAD2020_INTERMEDIATE_FILE_READER_H

#include "memory"
#include <fstream>
#include <string>

#include "../graph.h"
#include "../specs/timing_spec.h"


using namespace  std;


class IntermediateFileReader {


public:
    IntermediateFileReader(Graph& g, TimingSpec& timing_spec):
        g(g), timing_spec(timing_spec) {};

    void read(char*);
    void summary();

    void read_vlib();
    string read_vlib_common(vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>>&);
    void read_vlib_primitive();
    void read_vlib_module();
    void read_vlib_table(vector<string>&);
    void read_sdf();
    Graph& g;
    TimingSpec& timing_spec;

    ifstream fin;
};

#endif //ICCAD2020_INTERMEDIATE_FILE_READER_H
