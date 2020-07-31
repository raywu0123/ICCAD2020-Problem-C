#ifndef ICCAD2020_UTILS_H
#define ICCAD2020_UTILS_H

#include <string>
#include <constants.h>

double get_timescale(int num, const std::string& unit);

void cudaErrorCheck(cudaError_t);

#endif
