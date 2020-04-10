#ifndef ICCAD2020_INPUT_WAVEFORMS_H
#define ICCAD2020_INPUT_WAVEFORMS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "constants.h"
#include "circuit_model/circuit.h"

using namespace std;


class InputWaveforms {

public:
    int timescale_num;
    string timescale_unit;

    vector<pair<Timestamp, vector<pair<string, string>>>> dumps;
    unordered_map<string, pair<string, BitWidth>> token_to_wire;
};


#endif //ICCAD2020_INPUT_WAVEFORMS_H
