#include <iostream>
#include <vector>

#include "simulation_result.h"


using namespace std;


void SimulationResult::write(char *path) {
    f_out = ofstream(path);
}

VCDResult::VCDResult(
    Circuit& circuit, vector<string>& scopes, pair<int, string>& timescale_pair
): SimulationResult(circuit, scopes, timescale_pair) {}

void VCDResult::write(char *path) {
    SimulationResult::write(path);
    f_out << "$timescale " << timescale_pair.first << timescale_pair.second << " $end" << endl;
    for (const auto& scope : scopes) {
        f_out << "$scope module " << scope << " $end" << endl;
    }
    for (const auto& scope: scopes) {
        f_out << "$upscope" << endl;
    }
    merge_sort();
}

void VCDResult::merge_sort() {
    vector<unsigned int> indices;
    const auto num_wires = accumulators.size();
    unsigned int num_finished = 0;
    for (auto& acc: accumulators) {
        if (acc->transitions.empty()) num_finished++;
    }
    indices.resize(num_wires);
    while (num_finished < num_wires) {
        unsigned int min_index;
        Timestamp min_timestamp = LONG_LONG_MAX;
        for (int i = 0; i < num_wires; i++) {
            if(indices[i] >= accumulators[i]->transitions.size()) continue;
            const auto& transition = accumulators[i]->transitions[indices[i]];
            if (transition.timestamp < min_timestamp) {
                min_timestamp = transition.timestamp;
                min_index = i;
            }
        }
        bus_manager.add_transition(min_index, accumulators[min_index]->transitions[indices[min_index]]);
        indices[min_index]++;
        if (indices[min_index] >= accumulators[min_index]->transitions.size()) num_finished++;
    }
}

SAIFResult::SAIFResult(
    Circuit& circuit, vector<string>& scopes, pair<int, string>& timescale_pair
) : SimulationResult(circuit, scopes, timescale_pair) {}

void SAIFResult::write(char *path) {
    SimulationResult::write(path);
}
