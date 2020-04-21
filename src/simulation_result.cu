#include <iostream>
#include <vector>

#include "simulation_result.h"


using namespace std;


void SimulationResult::write(char *path) {
    f_out = ofstream(path);
}

VCDResult::VCDResult(
    const std::vector<Wire*>& wires, vector<string>& scopes, pair<int, string>& timescale_pair
): SimulationResult(wires, scopes, timescale_pair) {}

void VCDResult::write(char *path) {
    SimulationResult::write(path);
    f_out << "$timescale " << timescale_pair.first << timescale_pair.second << " $end" << endl;
    for (const auto& scope : scopes) {
        f_out << "$scope module " << scope << " $end" << endl;
    }
    for (const auto& scope: scopes) {
        f_out << "$upscope" << endl;
    }

    vector<pair<unsigned int, unsigned int>> buffer;
    vector<Timestamp> timestamps;
    merge_sort(buffer, timestamps);

    vector<pair<Timestamp, int>> timestamp_groups;
    group_timestamps(timestamps, timestamp_groups);

    int buffer_index = 0;
    for (const auto& group : timestamp_groups) {
        const auto& timestamp = group.first;
        cout << timestamp << endl;
        const auto& num_transitions = group.second;
        for (int i = 0; i < num_transitions; i++) {
//            TODO
            buffer_index++;
        }
    }
}

void VCDResult::merge_sort(vector<pair<unsigned int, unsigned int>>& buffer, vector<Timestamp>& timestamps) {
    vector<unsigned int> indices;
    const auto num_wires = wires.size();
    unsigned int num_finished = 0;
    for (auto& wire: wires) {
        const auto* acc = (VCDAccumulator*) wire->accumulator;
        if (acc->transitions.empty()) num_finished++;
    }
    indices.resize(num_wires);
    while (num_finished < num_wires) {
        unsigned int min_index;
        Timestamp min_timestamp = LONG_LONG_MAX;
        for (int i = 0; i < num_wires; i++) {
            const auto* accumulator = (VCDAccumulator*) wires[i]->accumulator;
            const auto& transitions = accumulator->transitions;
            if(indices[i] >= transitions.size()) continue;
            const auto& transition = transitions[indices[i]];
            if (transition.timestamp < min_timestamp) {
                min_timestamp = transition.timestamp;
                min_index = i;
            }
        }
        const auto* min_accumulator = (VCDAccumulator*) wires[min_index]->accumulator;
        const auto& transitions = min_accumulator->transitions;

        buffer.emplace_back(min_index, indices[min_index]);
        timestamps.push_back(transitions[indices[min_index]].timestamp);

        indices[min_index]++;
        if (indices[min_index] >= transitions.size()) num_finished++;
    }
}

void VCDResult::group_timestamps(const vector<Timestamp>& timestamps, vector<pair<Timestamp, int>>& timestamps_groups) {
    unsigned int size = timestamps.size();
    int group_size = 0;
    for (int i = 0; i < size; i++) {
        group_size++;
        if (i == size - 1 or timestamps[i + 1] != timestamps[i]) {
            timestamps_groups.emplace_back(timestamps[i], group_size);
            group_size = 0;
        }
    }
}

SAIFResult::SAIFResult(
    const std::vector<Wire*>& wires, vector<string>& scopes, pair<int, string>& timescale_pair
) : SimulationResult(wires, scopes, timescale_pair) {}

void SAIFResult::write(char *path) {
    SimulationResult::write(path);
}
