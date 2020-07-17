#ifndef ICCAD2020_UTILS_H
#define ICCAD2020_UTILS_H

#include <string>

double get_timescale(int num, const std::string& unit);
__host__ __device__ char get_edge_type(const char& v1, const char& v2);
#endif
