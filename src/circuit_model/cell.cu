#include <iostream>
#include <cassert>
#include <utility>
#include "cell.h"
#include "utils.h"

using namespace std;


Cell::Cell(
    const ModuleSpec* module_spec,
    const StdCellDeclare* declare,
    const WireMap<Wire>& pin_specs,
    Wire* supply1_wire, Wire* supply0_wire,
    string name
) : module_spec(module_spec), name(std::move(name)), num_args(declare->num_args)
{
    cudaMalloc((void**) &overflow_ptr, sizeof(bool));
    build_wire_map(declare, pin_specs, supply1_wire, supply0_wire);
}

Cell::~Cell() {
    cudaFree(overflow_ptr);
    for (auto& cell_wire : cell_wires) delete cell_wire->wire;
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

void Cell::build_wire_map(
    const StdCellDeclare* declare,
    const WireMap<Wire>& pin_specs,
    Wire *supply1_wire, Wire *supply0_wire
) {
    if (declare->num_args > MAX_NUM_MODULE_ARGS) {
        throw runtime_error("Too many module args (" + to_string(declare->num_args) + ")\n");
    }
    for (const auto& arg: declare->buckets[STD_CELL_INPUT]) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        auto* scheduled_wire = new ScheduledWire(wire_ptr);
        wire_map.set(arg, scheduled_wire); input_wires.push_back(scheduled_wire);
    }
    for (const auto& arg: declare->buckets[STD_CELL_SUPPLY1]) wire_map.set(arg, new IndexedWire(supply1_wire));
    for (const auto& arg: declare->buckets[STD_CELL_SUPPLY0]) wire_map.set(arg,  new IndexedWire(supply0_wire));

    for (const auto& arg: declare->buckets[STD_CELL_OUTPUT]) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        auto* indexed_wire = new IndexedWire(wire_ptr, capacity);
        wire_map.set(arg, indexed_wire); output_wires.push_back(indexed_wire);
    }
    for (const auto& arg: declare->buckets[STD_CELL_WIRE]) {
        auto* indexed_wire = new IndexedWire(new Wire(), capacity);
        wire_map.set(arg, indexed_wire); cell_wires.push_back(indexed_wire);
    }
    for (unsigned int arg = 0; arg < num_args; arg++) {
        if (wire_map.get(arg) == nullptr) cerr << "|WARNING: Arg (" + to_string(arg) + ") not found in wiremap of cell " << name  << endl;
    }
}


void Cell::init() {
    Cell::build_bucket_index_schedule(input_wires, INITIAL_CAPACITY - 1);
}

void Cell::build_bucket_index_schedule(vector<ScheduledWire*>& wires, unsigned int capacity) {
    unsigned int num_finished = 0, num_inputs = wires.size();

    vector<unsigned int> starting_indices; starting_indices.resize(num_inputs);
    vector<bool> finished; finished.resize(num_inputs);
    for (int i_wire = 0; i_wire < num_inputs; i_wire++) {
        const auto& wire = wires[i_wire];
        if (capacity >= wire->wire->bucket.size()) {
            finished[i_wire] = true;
            num_finished++;
        }
    }

    while (num_finished < num_inputs) {
//        Find min_end_timestamp
        Timestamp min_end_timestamp = LONG_LONG_MAX;
        for(int i_wire = 0; i_wire < num_inputs; i_wire++) {
            auto& wire = wires[i_wire];
            const auto& bucket = wire->wire->bucket;
            unsigned int end_index = starting_indices[i_wire] + capacity - 1;
            if (end_index >= bucket.size()) continue;
            const auto& end_timestamp = bucket.transitions[end_index].timestamp;
            if (end_timestamp < min_end_timestamp) min_end_timestamp = end_timestamp;
        }

        for (int i_wire = 0; i_wire < num_inputs; i_wire++) {
            auto& wire = wires[i_wire];
            const auto& bucket = wire->wire->bucket;
            const auto& bucket_size = bucket.size();
//            If already finished, push_back the last index of bucket
            if (not wire->bucket_index_schedule.empty() and wire->bucket_index_schedule.back() == bucket_size)
                wire->push_back_schedule_index(bucket_size);
            else {
//                FIXME will fail if start_index = 0 and timestamp[0] > min_end_timestamp
                auto start_index = bucket.transitions[starting_indices[i_wire]].timestamp > min_end_timestamp ? starting_indices[i_wire] - 1 : starting_indices[i_wire];
                auto end_index = find_end_index(bucket, start_index, min_end_timestamp, capacity);
                auto next_start_index = end_index + 1;
                wire->push_back_schedule_index(next_start_index);
                if (next_start_index + capacity >= bucket.size() and not finished[i_wire]) {
                    finished[i_wire] = true;
                    num_finished++;
                }
                starting_indices[i_wire] = end_index + 1;
            }
        }
    }
    for (auto& wire : wires) {
        wire->push_back_schedule_index(wire->wire->bucket.size());
    }
}

unsigned int Cell::find_end_index(const Bucket& bucket, unsigned int start_index, const Timestamp& t, unsigned int capacity) {
//    Binary Search for end_index <= t
    unsigned int low = start_index, high = min(start_index + capacity, bucket.size()) - 1;
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

void Cell::prepare_resource(int session_id, ResourceBuffer& resource_buffer)  {
    cudaMemset(overflow_ptr, 0, sizeof(bool)); // reset overflow value
    resource_buffer.overflows.push_back(overflow_ptr);
    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.sdf_specs.push_back(sdf_spec);
    resource_buffer.data_schedule_offsets.push_back(resource_buffer.data_schedule.size());
    resource_buffer.capacities.push_back(capacity);

    for (auto& indexed_wire : input_wires) indexed_wire->load(session_id);
    for (auto& indexed_wire : output_wires) indexed_wire->load(session_id);
    for (auto& indexed_wire : cell_wires) indexed_wire->load(session_id);

    for (unsigned int arg = 0; arg < num_args; ++arg) {
        const auto* indexed_wire = wire_map.get(arg);
        assert(indexed_wire != nullptr);
        if (indexed_wire->first_free_data_ptr_index - 1 >= indexed_wire->data_ptrs.size())
            throw runtime_error("Invalid access to indexed_wire's data_ptrs");
        resource_buffer.data_schedule.push_back(
            indexed_wire->data_ptrs[indexed_wire->first_free_data_ptr_index - 1]
        );
    }
}

void Cell::dump_result() {
    for (const auto& indexed_wire : output_wires) indexed_wire->store_to_bucket();
    if (finished()) {
        for (auto& indexed_wire : input_wires) indexed_wire->free();
        for (auto& indexed_wire : output_wires) indexed_wire->free();
        for (auto& indexed_wire : cell_wires) indexed_wire->free();
    }
}

bool Cell::finished() const {
    for (const auto& indexed_wire : input_wires) {
        if (not indexed_wire->finished()) return false;
    }
    return true;
}

bool Cell::overflow() const {
    bool host_overflow_value;
    cudaMemcpy(&host_overflow_value, overflow_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
    return host_overflow_value;
}

void Cell::handle_overflow() {
    capacity *= 2;
    for(auto& indexed_wire : input_wires) indexed_wire->handle_overflow();
    for(auto& indexed_wire : cell_wires) indexed_wire->handle_overflow();
    for(auto& indexed_wire : output_wires) indexed_wire->handle_overflow();
}


