#ifndef ICCAD2020_INTERMEDIATE_FILE_READER_H
#define ICCAD2020_INTERMEDIATE_FILE_READER_H

#include "memory"
#include <fstream>
#include <string>

#include "../circuit_model/circuit.h"
#include "../specs/timing_spec.h"
#include "../constants.h"


using namespace  std;


class IntermediateFileReader {


public:
    IntermediateFileReader(Circuit& circuit, TimingSpec& timing_spec):
        circuit(circuit), timing_spec(timing_spec) {};

    void read(char*);
    void summary() const;

    void read_vlib();
    string read_vlib_common(vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>>&);
    void read_vlib_primitive();
    void read_vlib_module();
    void read_vlib_table(vector<string>&);
    void read_sdf();

    Circuit& circuit;
    TimingSpec& timing_spec;

    ifstream fin;
};

#endif //ICCAD2020_INTERMEDIATE_FILE_READER_H
