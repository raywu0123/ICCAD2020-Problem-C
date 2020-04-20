#include <iostream>
#include <vector>

#include "simulation_result.h"


using namespace std;


void SimulationResult::write(char *path) {
    f_out = ofstream(path);
}

VCDResult::VCDResult(
    Circuit& circuit, vector<string>& scopes, pair<int, string>& timescale_pair
): SimulationResult(circuit, scopes, timescale_pair) {
    for (const auto& it : circuit.wires) {
        const auto& accumulator = new VCDAccumulator();
        it->accumulator = accumulator;
        accumulators.push_back(accumulator);
    }
}

void VCDResult::write(char *path) {
    SimulationResult::write(path);
    f_out << "$timescale " << timescale_pair.first << timescale_pair.second << " $end" << endl;
    for (const auto& scope : scopes) {
        f_out << "$scope module " << scope << " $end" << endl;
    }
    for (const auto& scope: scopes) {
        f_out << "$upscope" << endl;
    }
    const auto& buffer = merge_sort();
}

vector<Transition *> VCDResult::merge_sort() {
    vector<unsigned int> indices;
    const auto num_wires = circuit.wires.size();
    unsigned int num_finished = 0;
    for (auto& acc: accumulators) {
        if (acc->transitions.empty()) num_finished++;
    }
    indices.resize(num_wires);

    vector<Transition*> buffer;
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
        buffer.push_back(&(accumulators[min_index]->transitions[indices[min_index]]));
        indices[min_index]++;
        if (indices[min_index] >= accumulators[min_index]->transitions.size()) num_finished++;
    }
    return buffer;
}

SAIFResult::SAIFResult(
    Circuit& circuit, vector<string>& scopes, pair<int, string>& timescale_pair
) : SimulationResult(circuit, scopes, timescale_pair) {
    for (const auto& it : circuit.wires) {
        const auto& accumulator = new SAIFAccumulator();
        it->accumulator = accumulator;
        accumulators.push_back(accumulator);
    }
}

void SAIFResult::write(char *path) {
    SimulationResult::write(path);
}
