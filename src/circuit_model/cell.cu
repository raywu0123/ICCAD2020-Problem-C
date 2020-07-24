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
) : module_spec(module_spec), name(std::move(name))
{
    num_args = declare->num_input + declare->num_output;
    build_wire_map(declare, pin_specs, supply1_wire, supply0_wire);
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
    if (num_args > MAX_NUM_MODULE_ARGS) {
        throw runtime_error("Too many module args (" + to_string(num_args) + ")\n");
    }
    for (unsigned int arg = 0; arg < declare->num_input; ++arg) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        auto* scheduled_wire = new ScheduledWire(wire_ptr);
        wire_map.set(arg, scheduled_wire); input_wires.push_back(scheduled_wire);
    }
    for (unsigned int arg = declare->num_input; arg < num_args; ++arg) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        auto* indexed_wire = new IndexedWire(wire_ptr);
        wire_map.set(arg, indexed_wire); output_wires.push_back(indexed_wire);
    }
    for (unsigned int arg = 0; arg < num_args; arg++) {
        if (wire_map.get(arg) == nullptr) cerr << "| WARNING: Arg (" + to_string(arg) + ") not found in wiremap of cell " << name  << endl;
    }
}

void Cell::prepare_resource(int session_id, ResourceBuffer& resource_buffer)  {
    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.sdf_specs.push_back(sdf_spec);
    resource_buffer.data_schedule_offsets.push_back(resource_buffer.data_schedule.size());

    for (auto& indexed_wire : input_wires) indexed_wire->load(session_id);
    for (auto& indexed_wire : output_wires) indexed_wire->load(session_id);

    for (unsigned int arg = 0; arg < num_args; ++arg) {
        const auto* indexed_wire = wire_map.get(arg);
        assert(indexed_wire != nullptr);
        if (indexed_wire->first_free_data_ptr_index - 1 >= indexed_wire->data_ptrs.size())
            throw runtime_error("Invalid access to indexed_wire's data_ptrs");
        resource_buffer.data_schedule.push_back(
            indexed_wire->data_ptrs[indexed_wire->first_free_data_ptr_index - 1]
        );
        resource_buffer.progress_updates.push_back(indexed_wire->progress_update_ptr);
    }
}

void Cell::gather_results() {
    for (auto& wire : input_wires) wire->update_progress();
    for (const auto& indexed_wire : output_wires) indexed_wire->store_to_bucket();
    if (finished()) {
        for (auto& indexed_wire : input_wires) indexed_wire->free();
        for (auto& indexed_wire : output_wires) indexed_wire->free();
    }
}

bool Cell::finished() const {
    for (const auto& indexed_wire : input_wires) {
        if (not indexed_wire->finished()) return false;
    }
    return true;
}
