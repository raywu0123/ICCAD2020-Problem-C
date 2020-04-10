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
    explicit IntermediateFileReader(TimingSpec& timing_spec, ifstream& fin): timing_spec(timing_spec), fin(fin) {};
    void summary() const;
    void read_sdf();

private:


    TimingSpec& timing_spec;
    ifstream& fin;
};

#endif
