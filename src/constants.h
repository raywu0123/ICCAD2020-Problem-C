#ifndef ICCAD2020_CONSTANTS_H
#define ICCAD2020_CONSTANTS_H

#include <vector>
#include <string>

#include "simulator/data_structures.h"


const unsigned int N_GATE_PARALLEL = 32;
const unsigned int N_STIMULI_PARALLEL = 1024;
const unsigned int INITIAL_CAPACITY = 32;


const Wirekey SUPPLY1_WIREKEY = Wirekey{"1'b1", 0};
const Wirekey SUPPLY0_WIREKEY = Wirekey{"1'b0", 0};

const int NUM_VALUES = 4;
const char VALUES[NUM_VALUES] = {'0', '1', 'x', 'z'};

enum STD_CELL_DECLARE_TYPE {
    STD_CELL_INPUT,
    STD_CELL_OUTPUT,
    STD_CELL_WIRE,
    STD_CELL_SUPPLY1,
    STD_CELL_SUPPLY0,
    STD_CELL_LAST=STD_CELL_SUPPLY0,
};

const STD_CELL_DECLARE_TYPE STD_CELL_DECLARE_TYPES[] = {
    STD_CELL_INPUT, STD_CELL_OUTPUT, STD_CELL_WIRE, STD_CELL_SUPPLY1, STD_CELL_SUPPLY0
};

#endif
