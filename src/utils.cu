#include <iostream>
#include <string>

#include "utils.h"

using namespace std;


double get_timescale(int num, const string& unit) {
    double unit_num;
    if (unit == "ms") {
        unit_num = 1e-6;
    } else if (unit == "ns") {
        unit_num = 1e-9;
    } else if (unit == "ps") {
        unit_num = 1e-12;
    } else throw runtime_error("Unrecognized timescale unit: " + unit + "\n");

    return num * unit_num;
}

__host__ __device__ char get_edge_type(const char& v1, const char& v2) {
    if (v2 == '1' or v1 == '0') return '+';
    if (v2 == '0' or v1 == '1') return '-';
    return 'x';
}

void cudaErrorCheck(cudaError_t status) {
    if (status != cudaSuccess) throw runtime_error(cudaGetErrorString(status));
}
