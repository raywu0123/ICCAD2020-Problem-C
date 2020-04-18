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
