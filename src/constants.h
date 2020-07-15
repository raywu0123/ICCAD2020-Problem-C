#ifndef ICCAD2020_CONSTANTS_H
#define ICCAD2020_CONSTANTS_H

#include "simulator/data_structures.h"

const unsigned int N_CELL_PARALLEL = 1024;
const unsigned int N_STIMULI_PARALLEL = 256;
const unsigned int INITIAL_CAPACITY = 16;
const unsigned int MAX_NUM_MODULE_OUTPUT = 10;
const unsigned int MAX_DATA_SCHEDULE_SIZE = 30;

const Wirekey SUPPLY1_WIREKEY = Wirekey{"1'b1", 0};
const Wirekey SUPPLY0_WIREKEY = Wirekey{"1'b0", 0};

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
