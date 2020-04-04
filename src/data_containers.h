#ifndef ICCAD2020_DATA_CONTAINERS_H
#define ICCAD2020_DATA_CONTAINERS_H

#include <string>
#include <vector>

#include "constants.h"

using namespace std;


class Signal {
//    2D char array with shape (n_stimuli, bitwidth)
public:
    explicit Signal(BitWidth bitwidth): bitwidth(std::move(bitwidth)) {};

    char** data = nullptr;
    BitWidth bitwidth;
};

#endif //ICCAD2020_DATA_CONTAINERS_H
