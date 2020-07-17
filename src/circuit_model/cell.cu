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
    build_wire_map(declare, pin_specs, supply1_wire, supply0_wire);
}

Cell::~Cell() {
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
        auto* indexed_wire = new IndexedWire(wire_ptr);
        wire_map.set(arg, indexed_wire); output_wires.push_back(indexed_wire);
    }
    for (const auto& arg: declare->buckets[STD_CELL_WIRE]) {
        auto* indexed_wire = new IndexedWire(new Wire());
        wire_map.set(arg, indexed_wire); cell_wires.push_back(indexed_wire);
    }
}


void Cell::init() {
    build_scheduled_buckets(input_wires, starting_indices);
}

void Cell::build_scheduled_buckets(vector<ScheduledWire*>& wires, vector<unsigned int>& starting_indices) {
    // initialize indices, num_finished
    unsigned int num_finished = 0; const auto& num_wires = wires.size();

    unsigned int sum_size = 0;
    for (const auto& wire : wires) {
        const auto& size = wire->wire->bucket.size();
        sum_size += size;
        if (size <= 1) num_finished++;
    }
    for (auto& wire : wires) {
        const auto& transitions = wire->wire->bucket.transitions;
        wire->scheduled_bucket.reserve(sum_size);
        wire->scheduled_bucket.push_back(transitions.front());
    }
    starting_indices.reserve(sum_size);
    vector<unsigned int> indices; indices.resize(num_wires);

    // merge sort
    while (num_finished < num_wires) {
        // find min timestamp and corresponding wire index
        Timestamp min_t = LONG_LONG_MAX;
        for (unsigned int i = 0; i < num_wires; i++) {
            if (indices[i] + 1 >= wires[i]->size()) continue;
            const auto& t = wires[i]->wire->bucket[indices[i] + 1].timestamp;
            if (t < min_t) min_t = t;
        }
        assert(min_t != LONG_LONG_MAX);
        starting_indices.push_back(wires.front()->scheduled_bucket.size());

        vector<unsigned int> advancing; advancing.reserve(num_wires);
        for (unsigned int i = 0; i < num_wires; i++) {
            const auto& b = wires[i]->wire->bucket;
            auto& index = indices[i];
            if (b[index + 1].timestamp == min_t) {
                advancing.push_back(i);
                index += 1;
                if (index + 1 >= b.size()) num_finished++;
            }
        }
        for(const auto& advancing_wire_index : advancing) {
            const auto& advancing_inner_bucket = wires[advancing_wire_index]->wire->bucket;
            auto edge_type = get_edge_type(
                advancing_inner_bucket[indices[advancing_wire_index] - 1].value,
                advancing_inner_bucket[indices[advancing_wire_index]].value
            );
            DelayInfo d{advancing_wire_index, edge_type};
            for (unsigned int i = 0; i < num_wires; i++) {
                auto& wire = wires[i];
                wire->scheduled_bucket.emplace_back(min_t, wire->wire->bucket[indices[i]].value, d);
            }
        }
    }
    starting_indices.push_back(wires.front()->scheduled_bucket.size());
}

void Cell::prepare_resource(int session_id, ResourceBuffer& resource_buffer)  {
    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.sdf_specs.push_back(sdf_spec);
    resource_buffer.data_schedule_offsets.push_back(resource_buffer.data_schedule.size());
    resource_buffer.capacities.push_back(capacity);

    unsigned int progress = 0;
    for (auto& indexed_wire : input_wires) progress = indexed_wire->load(session_id, starting_indices, progress_index);
    for (auto& indexed_wire : output_wires) indexed_wire->load(session_id);
    for (auto& indexed_wire : cell_wires) indexed_wire->load(session_id);
    progress_index = progress;

    for (unsigned int arg = 0; arg < num_args; ++arg) {
        const auto& indexed_wire = wire_map.get(arg);
        if (indexed_wire->first_free_data_ptr_index - 1 >= indexed_wire->data_ptrs.size())
            throw runtime_error("Invalid access to indexed_wire's data_ptrs");
        resource_buffer.data_schedule.push_back(indexed_wire->data_ptrs[indexed_wire->first_free_data_ptr_index - 1]);
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
    return progress_index >= starting_indices.size() - 1;
}

