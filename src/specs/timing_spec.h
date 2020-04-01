#ifndef ICCAD2020_TIMING_SPEC_H
#define ICCAD2020_TIMING_SPEC_H

#include "memory"
#include <vector>
#include "sdf_cell.h"

using namespace  std;

class TimingSpec {
public:
    TimingSpec() = default;

    vector<shared_ptr<SDFCell>> cells;
    int timescale_num{};
    string timescale_unit;
};

#endif //ICCAD2020_TIMING_SPEC_H