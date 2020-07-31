#ifndef ICCAD2020_CONSTANTS_H
#define ICCAD2020_CONSTANTS_H

#include "simulator/data_structures.h"

const unsigned int N_THREAD = 8;
const unsigned int N_CELL_PER_THREAD = 1024;
const unsigned int N_CELL_PARALLEL = N_THREAD * N_CELL_PER_THREAD;
const unsigned int N_STIMULI_PARALLEL = 256;
const unsigned int INITIAL_CAPACITY = 16;
const unsigned int MAX_NUM_MODULE_OUTPUT = 10;
const unsigned int MAX_NUM_MODULE_ARGS = 20;

#endif
