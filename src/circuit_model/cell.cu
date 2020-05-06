#include <iostream>

#include "cell.h"

using namespace std;


Cell::Cell(
    const ModuleSpec* module_spec,
    const vector<SubmoduleSpec>* submodule_specs,
    const StdCellDeclare* declare,
    const vector<PinSpec> &pin_specs,
    Wire* supply1_wire, Wire* supply0_wire,
    const vector<Wire*>& alloc_wires_param, const vector<Wire*>& free_wires_param
) : module_spec(module_spec)
{
    build_wire_map(declare, pin_specs, supply1_wire, supply0_wire);
    create_wire_schedule(submodule_specs);
    for (const auto& idx : declare->buckets[STD_CELL_INPUT]) {
        input_wires.emplace_back(wire_map[idx]);
    }
    for (const auto& idx : declare->buckets[STD_CELL_OUTPUT]) {
        output_wires.push_back(wire_map[idx]);
    }
}

void Cell::set_paths(const vector<SDFPath>& ps) {
    vector<char> edge_types;
    vector<unsigned int> input_indices, output_indices;
    vector<int> rising_delays, falling_delays;

    for (const auto& path : ps) {
        input_indices.push_back(path.in);
        output_indices.push_back(path.out);
        edge_types.push_back(path.edge_type);
        rising_delays.push_back(path.rising_delay);
        falling_delays.push_back(path.falling_delay);
    }

    SDFSpec host_sdf_spec{};
    auto num_rows = ps.size();
    host_sdf_spec.num_rows = num_rows;
    cudaMalloc((void**) &host_sdf_spec.edge_type, sizeof(char) * num_rows);
    cudaMalloc((void**) &host_sdf_spec.input_index, sizeof(int) * num_rows);
    cudaMalloc((void**) &host_sdf_spec.output_index, sizeof(int) * num_rows);
    cudaMalloc((void**) &host_sdf_spec.rising_delay, sizeof(int) * num_rows);
    cudaMalloc((void**) &host_sdf_spec.falling_delay, sizeof(int) * num_rows);
    cudaMemcpy(host_sdf_spec.edge_type, edge_types.data(), sizeof(char) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(host_sdf_spec.input_index, input_indices.data(), sizeof(int) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(host_sdf_spec.output_index, output_indices.data(), sizeof(int) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(host_sdf_spec.rising_delay, rising_delays.data(), sizeof(int) * num_rows, cudaMemcpyHostToDevice);
    cudaMemcpy(host_sdf_spec.falling_delay, falling_delays.data(), sizeof(int) * num_rows, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &sdf_spec, sizeof(SDFSpec));
    cudaMemcpy(sdf_spec, &host_sdf_spec, sizeof(SDFSpec), cudaMemcpyHostToDevice);
}

Cell::~Cell() {
    for (auto& wire_ptr: cell_wires) {
        delete wire_ptr;
    }
}

void Cell::build_wire_map(
    const StdCellDeclare* declare,
    const vector<PinSpec> &pin_specs,
    Wire *supply1_wire, Wire *supply0_wire
)
{
    if (not wire_map.empty()) throw runtime_error("wire_map not empty.");

    for (const auto& pin_spec: pin_specs) wire_map[pin_spec.index] = pin_spec.wire;
    for (const auto& arg: declare->buckets[STD_CELL_SUPPLY1]) wire_map[arg] = supply1_wire;
    for (const auto& arg: declare->buckets[STD_CELL_SUPPLY0]) wire_map[arg] = supply0_wire;
}


void Cell::create_wire_schedule(
    const vector<SubmoduleSpec>* submodule_specs
)  {
    for(const auto& submodule_spec: *submodule_specs) {
        for (const auto& arg: submodule_spec.args) {
            const auto& it = wire_map.find(arg);
            if (it != wire_map.end()) {
                wire_schedule.emplace_back(it->second);
            } else {
                // create cell wire
                auto* wire_ptr = new Wire();
                wire_schedule.emplace_back(wire_ptr);
                add_cell_wire(wire_ptr);
            }
        }
    }
}

void Cell::add_cell_wire(Wire *wire_ptr) {
    cell_wires.push_back(wire_ptr);
}

void Cell::build_bucket_index_schedule(vector<IndexedWire>& wires, unsigned int capacity) {
    unsigned int num_finished = 0;
    unsigned int num_inputs = wires.size();

    vector<unsigned int> starting_indices;
    vector<bool> finished;
    starting_indices.resize(num_inputs);
    finished.resize(num_inputs);
    for (int i_wire = 0; i_wire < num_inputs; i_wire++) {
        const auto& wire = wires[i_wire];
        if (capacity >= wire.wire->bucket.size()) {
            finished[i_wire] = true;
            num_finished++;
        }
    }

    while (num_finished < num_inputs) {
//        Find min_end_timestamp
        Timestamp min_end_timestamp = LONG_LONG_MAX;
        for(int i_wire = 0; i_wire < num_inputs; i_wire++) {
            auto& wire = wires[i_wire];
            const auto& bucket = wire.wire->bucket;
            unsigned int end_index = starting_indices[i_wire] + capacity - 1;
            if (end_index >= bucket.size()) continue;
            const auto& end_timestamp = bucket.transitions[end_index].timestamp;
            if (end_timestamp < min_end_timestamp) min_end_timestamp = end_timestamp;
        }

        for (int i_wire = 0; i_wire < num_inputs; i_wire++) {
            auto& wire = wires[i_wire];
            const auto& bucket = wire.wire->bucket;
//            If already finished, push_back the last index of bucket
            if (
                not wire.bucket_index_schedule.empty()
                and wire.bucket_index_schedule.back() == bucket.size()
            ) wire.push_back_schedule_index(bucket.size() - 1);
            else {
//                FIXME will fail if start_index = 0 and timestamp[0] > min_end_timestamp
                auto start_index = bucket.transitions[starting_indices[i_wire]].timestamp > min_end_timestamp ? starting_indices[i_wire] - 1 : starting_indices[i_wire];
                auto end_index = find_end_index(bucket, start_index, min_end_timestamp, capacity);
                auto next_start_index = end_index + 1;
                wire.push_back_schedule_index(next_start_index);
                if (next_start_index + capacity >= bucket.size() and not finished[i_wire]) {
                    finished[i_wire] = true;
                    num_finished++;
                }
                starting_indices[i_wire] = end_index + 1;
            }
        }
    }
    for (auto& wire : wires) {
        wire.push_back_schedule_index(wire.wire->bucket.size());
    }
}

unsigned int Cell::find_end_index(const Bucket& bucket, unsigned int start_index, Timestamp t, unsigned int capacity) {
//    Binary Search for end_index <= t
    unsigned int low = start_index;
    unsigned int high = min(start_index + capacity, bucket.size()) - 1;
    if (bucket.transitions[high].timestamp <= t) return high;
    while (low < high) {
        unsigned mid = (low + high) / 2;
        if (mid == low) break;
        if (bucket.transitions[mid].timestamp < t) low = mid;
        else if (bucket.transitions[mid].timestamp > t) high = mid;
        else return mid;
    }
    return low;
}

bool Cell::prepare_resource(ResourceBuffer& resource_buffer)  {
    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.sdf_specs.push_back(sdf_spec);
    resource_buffer.data_schedule_offsets.push_back(resource_buffer.data_schedule_offsets.size());

//    allocate data memory
    for (const auto& wire : wire_schedule) {
        auto* data_ptr = wire->alloc();
        resource_buffer.data_schedule.emplace_back(data_ptr, wire->capacity);
    }

    bool all_finished = true;
    for (auto& indexed_wire : input_wires) {
        all_finished &= indexed_wire.load_from_bucket();
    }
    return all_finished;
}

void Cell::finalize() {
    for (const auto& wire : output_wires) {
        wire->store_to_bucket();
    }
    for (const auto& wire : wire_schedule) {
        wire->free();
    }
}

