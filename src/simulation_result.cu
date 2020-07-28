#include <iostream>
#include <vector>
#include <queue>
#include <ctime>
#include <chrono>

#include "simulation_result.h"


using namespace std;


void SimulationResult::write(char *path) {
    f_out = ofstream(path);
}

VCDResult::VCDResult(
    const std::vector<Wire*>& wires,
    vector<string>& scopes,
    pair<int, string>& timescale_pair,
    Timestamp dumpon_time, Timestamp dumpoff_time,
    BusManager& bus_manager
): SimulationResult(wires, scopes, timescale_pair, dumpon_time, dumpoff_time, bus_manager) {}

void VCDResult::write(char *path) {
    SimulationResult::write(path);
    f_out << "$timescale " << timescale_pair.first << timescale_pair.second << " $end" << endl;
    for (const auto& scope : scopes) {
        f_out << "$scope module " << scope << " $end" << endl;
    }
    f_out << bus_manager.dumps_token_to_bus_map();
    for (const auto& scope: scopes) {
        f_out << "$upscope" << endl;
    }
    f_out << "$enddefinitions $end" << endl;
    f_out << "$dumpvars" << endl;

    vector<pair<unsigned int, unsigned int>> buffer;
    vector<Timestamp> timestamps;

    auto& f_wires = wires;
    merge_sort(f_wires, buffer, timestamps, dumpon_time, dumpoff_time);

    vector<pair<Timestamp, int>> timestamp_groups;
    group_timestamps(timestamps, timestamp_groups);

    bus_manager.write_init(f_wires);
    f_out << bus_manager.dumps_result(0);

    int buffer_index = 0;
    Timestamp prev_timestamp = LONG_LONG_MIN;
    for (const auto& group : timestamp_groups) {
        const auto& timestamp = group.first;
        if (prev_timestamp < dumpon_time and timestamp >= dumpon_time and timestamp != 0) {
            bus_manager.dumpon_init(f_wires);
            if (timestamp > dumpon_time) f_out << bus_manager.dumps_result(dumpon_time);
        }
        prev_timestamp = timestamp;

        const auto& num_transitions = group.second;
        for (int i = 0; i < num_transitions; i++) {
            const auto& wire_index = buffer[buffer_index].first;
            const auto& transition_index = buffer[buffer_index].second;
            const auto& bucket = f_wires[wire_index]->bucket;
            const auto& transition = bucket.transitions[transition_index];
            const auto& wire_infos = f_wires[wire_index]->wire_infos;

            bus_manager.add_transition(wire_infos, transition);
            buffer_index++;
        }
        if (timestamp < dumpon_time or timestamp == 0) {
            // problem when constant wire with dumpon_time == 0
            bus_manager.dumps_result(timestamp);
        } else {
            f_out << bus_manager.dumps_result(timestamp);
        }
    }

    if (not timestamp_groups.empty() and timestamp_groups.back().first < dumpon_time) {
        bus_manager.dumpon_init(f_wires);
        bus_manager.dumps_result(dumpon_time);
    }
    f_out.close();
}

void VCDResult::merge_sort(
    const vector<Wire*>& f_wires,
    vector<pair<unsigned int, unsigned int>>& buffer,
    vector<Timestamp>& timestamps,
    Timestamp dumpon_time, Timestamp dumpoff_time
) {
    cout << "| Status: Merge sorting VCD result..." << endl;
    const auto num_wires = f_wires.size();
    vector<unsigned int> indices; indices.resize(num_wires);
    unsigned int sum_num_transitions = 0;
    for (unsigned int i = 0; i < num_wires; ++i) {
        const auto& wire = f_wires[i];
        const auto& bucket = wire->bucket;
        const auto& bucket_size = bucket.size();
        sum_num_transitions += bucket_size;
        indices[i] = binary_search(bucket.transitions.data(), bucket_size - 1, dumpon_time);
        indices[i] = indices[i] > 0 and bucket[indices[i]].timestamp >= dumpon_time ? indices[i] - 1: indices[i];
    }
    buffer.reserve(sum_num_transitions); timestamps.reserve(sum_num_transitions);

    priority_queue<PriorityQueueEntry> priority_queue;
    for (int i = 0; i < num_wires; i++) {
        const auto& wire = f_wires[i];
        const auto bucket = wire->bucket;
        if (indices[i] >= bucket.size()) continue;
        priority_queue.emplace(i, bucket[indices[i]].timestamp);
    }

    while (not priority_queue.empty()) {
        const auto top = priority_queue.top();
        if (top.t > dumpoff_time) break;
        buffer.emplace_back(top.bucket_index, indices[top.bucket_index]);
        timestamps.push_back(top.t);
        priority_queue.pop();

        indices[top.bucket_index]++;

        const auto& bucket = f_wires[top.bucket_index]->bucket;
        if (indices[top.bucket_index] < bucket.size()) priority_queue.emplace(top.bucket_index, bucket[indices[top.bucket_index]].timestamp);
    }
}

void VCDResult::group_timestamps(const vector<Timestamp>& timestamps, vector<pair<Timestamp, int>>& timestamps_groups) {
    cout << "| Status: Grouping timestamps..." << endl;
    unsigned int size = timestamps.size();
    timestamps_groups.reserve(size);
    int group_size = 0;
    for (int i = 0; i < size; i++) {
        group_size++;
        if (i == size - 1 or timestamps[i + 1] != timestamps[i]) {
            timestamps_groups.emplace_back(timestamps[i], group_size);
            group_size = 0;
        }
    }
}

void VCDResult::filter_wires(const vector<Wire*>& ws, vector<Wire*>& f_ws) {
    for (auto* w : ws) {
        if (w->wire_infos.empty()) continue;
        const auto& name = w->wire_infos[0].wirekey.first;
        if (name == "mcif2noc_axi_aw_awvalid" or name == "n60869" or name == "n78479"
        ) f_ws.push_back(w);
    }
}

SAIFResult::SAIFResult(
    const std::vector<Wire*>& wires, vector<string>& scopes, pair<int, string>& timescale_pair,
    Timestamp dumpon_time, Timestamp dumpoff_time,
    BusManager& bus_manager
) : SimulationResult(wires, scopes, timescale_pair, dumpon_time, dumpoff_time, bus_manager) {}

void SAIFResult::write(char *path) {
    SimulationResult::write(path);
    f_out << "(SAIFILE" << endl;
    f_out << "(SAIFVERSION \"2.0\")" << endl;
    f_out << "(DIRECTION \"backward\")" << endl;
    f_out << "(DESIGN )" << endl;
    std::time_t t_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    f_out << "(DATE \"" << strtok(std::ctime(&t_now), "\n") << "\")" << endl;
    f_out << "(VENDOR \"Synopsys, Inc\")" << endl;
    f_out << "(PROGRAM_NAME \"VCS K-2015.09-SP2_Full64\")" << endl;
    f_out << "(VERSION \"1.0\")" << endl;
    f_out << "(DIVIDER / )" << endl;
    f_out << "(TIMESCALE " << timescale_pair.first << " " << timescale_pair.second << ")" << endl;
    f_out << "(DURATION " << dumpoff_time - dumpon_time << ")" << endl;
    f_out << "(INSTANCE " << scopes[0] << endl;
    f_out << indent << "(INSTANCE " << scopes[1] << endl;
    f_out << indent << indent << "(NET" << endl;

    for (const auto& wire : wires) {
        const auto wire_stats = calculate_wire_stats(wire->bucket, dumpon_time, dumpoff_time);
        for (const auto& wireinfo : wire->wire_infos){
            const auto& bus = bus_manager.buses[wireinfo.bus_index];
            write_wirekey_result(bus.bitwidth, wireinfo.wirekey, wire_stats);
        }
    }

    f_out << indent << indent << ")" << endl; // NET
    f_out << indent << ")" << endl;  // Second INSTANCE
    f_out << ")" << endl;  // First INSTANCE
    f_out << ")" << endl;  // SAIFILE
}

void SAIFResult::write_wirekey_result(const BitWidth& bitwidth, const Wirekey &wirekey, const WireStat &wirestat) {
    f_out   << indent << indent << indent
            << "(" << wirekey.first;

    if (bitwidth.first != bitwidth.second) f_out   << "\\[" << wirekey.second << "\\]";
    f_out   << endl;

    f_out   << indent << indent << indent << indent
            << "(T0 " << wirestat.T0 << ") (T1 " << wirestat.T1 << ") (TX " << wirestat.TX << ")";
    if (wirestat.TZ != 0) f_out << " (TZ " << wirestat.TZ << ")";
    f_out   << endl;

    f_out   << indent << indent << indent
            << ")" << endl;
}

WireStat SAIFResult::calculate_wire_stats(const Bucket& bucket, Timestamp dumpon_time, Timestamp dumpoff_time) {
    WireStat wirestat{};
    const auto& transitions = bucket.transitions;
    const auto& size = transitions.size();
    for (unsigned idx = 1; idx < size; idx++) {
        const auto &t_curr = transitions[idx], &t_prev = transitions[idx - 1];
        if (t_curr.timestamp <= dumpon_time) continue;
        if (t_prev.timestamp >= dumpoff_time) break;

        auto d = min(t_curr.timestamp, dumpoff_time) - max(t_prev.timestamp, dumpon_time);
        auto& prev_v = t_prev.value;
        wirestat.update(prev_v, d);
    }

    const auto last_transition = transitions.back();
    if (last_transition.timestamp < dumpoff_time) {
        wirestat.update(last_transition.value, dumpoff_time - max(last_transition.timestamp, dumpon_time));
    }
    return wirestat;
}
