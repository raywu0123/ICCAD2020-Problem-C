#ifndef ICCAD2020_UTILS_H
#define ICCAD2020_UTILS_H

#include <string>
#include <simulator/data_structures.h>

double get_timescale(int num, const std::string& unit);


void cudaErrorCheck(cudaError_t);
#endif
