#ifndef ICCAD2020_UTILS_H
#define ICCAD2020_UTILS_H

#include <string>

double get_timescale(int num, const std::string& unit);
__host__ __device__ char get_edge_type(char v1, char v2);

#endif
