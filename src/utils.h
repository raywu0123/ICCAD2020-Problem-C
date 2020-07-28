#ifndef ICCAD2020_UTILS_H
#define ICCAD2020_UTILS_H

#include <string>
#include <simulator/data_structures.h>

double get_timescale(int num, const std::string& unit);

__host__ __device__ char get_edge_type(const Values& v1, const Values& v2);

void cudaErrorCheck(cudaError_t);
#endif
