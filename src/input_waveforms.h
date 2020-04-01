#ifndef ICCAD2020_INPUT_WAVEFORMS_H
#define ICCAD2020_INPUT_WAVEFORMS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "constants.h"
#include "graph.h"

using namespace std;


class InputWaveforms {

public:
    int timescale_num;
    string timescale_unit;

    unordered_map<string, shared_ptr<Wire>> token_to_wire;
    vector<
        pair<Timestamp, vector<pair<string, string>>>
    > dumps;
};


#endif //ICCAD2020_INPUT_WAVEFORMS_H
