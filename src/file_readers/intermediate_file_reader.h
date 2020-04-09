#ifndef ICCAD2020_INTERMEDIATE_FILE_READER_H
#define ICCAD2020_INTERMEDIATE_FILE_READER_H

#include "memory"
#include <fstream>
#include <string>

#include "../circuit_model/circuit.h"
#include "../specs/timing_spec.h"
#include "../constants.h"
#include "../simulator/module_registry.h"


using namespace  std;


class IntermediateFileReader {


public:
    IntermediateFileReader(Circuit& circuit, TimingSpec& timing_spec, ModuleRegistry& module_registry):
        circuit(circuit), timing_spec(timing_spec), module_registry(module_registry) {};

    void read(char*);
    void summary() const;

    void read_vlib();
    string read_vlib_common(StdCellDeclare&);
    void read_vlib_primitive();
    void read_vlib_module();
    void read_vlib_table(vector<string>&);
    void read_sdf();

    Circuit& circuit;
    TimingSpec& timing_spec;
    ModuleRegistry& module_registry;

    ifstream fin;
};

#endif //ICCAD2020_INTERMEDIATE_FILE_READER_H
