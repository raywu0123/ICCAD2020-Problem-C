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
    cudaMalloc((void**) &overflow_ptr, sizeof(bool));

    build_wire_map(declare, pin_specs, supply1_wire, supply0_wire);
    create_wire_schedule(submodule_specs);
    init_wire_vectors(declare);
}

void Cell::init_wire_vectors(const StdCellDeclare* declare) {
    for (const auto& idx : declare->buckets[STD_CELL_INPUT]) input_wires.emplace_back(wire_map[idx]);
    for (const auto& idx : declare->buckets[STD_CELL_OUTPUT]) output_wires.emplace_back(wire_map[idx]);

    for (auto& indexed_wire : input_wires) indexed_wire.init_wire_schedule_indices(wire_schedule);
    for (auto& indexed_wire : output_wires) indexed_wire.init_wire_schedule_indices(wire_schedule);
    for (auto& indexed_wire : cell_wires) indexed_wire.init_wire_schedule_indices(wire_schedule);
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
    cell_wires.emplace_back(wire_ptr);
}

void Cell::build_bucket_index_schedule(vector<ScheduledWire>& wires, unsigned int capacity) {
    unsigned int num_finished = 0;
    unsigned int num_inputs = wires.size();

    vector<unsigned int> starting_indices; starting_indices.resize(num_inputs);
    vector<bool> finished; finished.resize(num_inputs);
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
            ) wire.push_back_schedule_index(bucket.size());
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

void Cell::init() {
    for (auto& indexed_wire : input_wires) indexed_wire.wire->reset_capacity();

    Cell::build_bucket_index_schedule(input_wires, INITIAL_CAPACITY - 1);
    // leave one for delay calculation

    data_ptrs.resize(wire_schedule.size());
    //    allocate data memory
    for (auto& indexed_wire : input_wires) update_data_ptrs(indexed_wire.alloc(), indexed_wire);
    for (auto& indexed_wire : output_wires) update_data_ptrs(indexed_wire.alloc(), indexed_wire);
    for (auto& indexed_wire : cell_wires) update_data_ptrs(indexed_wire.alloc(), indexed_wire);

    for (auto& indexed_wire : input_wires) indexed_wire.load_from_bucket();
}

void Cell::update_data_ptrs(Transition* data_ptr, const IndexedWire& indexed_wire) {
    for (auto schedule_index : indexed_wire.wire_schedule_indices) {
        data_ptrs[schedule_index] = data_ptr;
    }
}

void Cell::prepare_resource(ResourceBuffer& resource_buffer)  {
    cudaMemset(&overflow_ptr, 0, sizeof(bool)); // reset overflow value
    resource_buffer.overflows.push_back(overflow_ptr);

    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.sdf_specs.push_back(sdf_spec);
    resource_buffer.data_schedule_offsets.push_back(resource_buffer.data_schedule_offsets.size());

    for (int i = 0; i < wire_schedule.size(); i++) {
        resource_buffer.data_schedule.emplace_back(data_ptrs[i], wire_schedule[i]->capacity);
    }
}

void Cell::dump_result() {
    for (const auto& indexed_wire : output_wires) {
        indexed_wire.wire->store_to_bucket();
    }
}

bool Cell::next() {
    dump_result();

    bool finished = true;
    for (auto& indexed_wire : input_wires) finished &= indexed_wire.next();

    if (finished) {
        for (auto& indexed_wire : input_wires) indexed_wire.free();
        for (auto& indexed_wire : output_wires) indexed_wire.free();
        for (auto& indexed_wire : cell_wires) indexed_wire.free();
    }
    else for (auto& indexed_wire : input_wires) indexed_wire.load_from_bucket();

    return finished;
}

void Cell::increase_capacity() {
    for (auto& indexed_wire : cell_wires) update_data_ptrs(indexed_wire.increase_capacity(), indexed_wire);
    for (auto& indexed_wire : output_wires) update_data_ptrs(indexed_wire.increase_capacity(), indexed_wire);
}

bool Cell::overflow() const {
    bool host_overflow_value;
    cudaMemcpy(&host_overflow_value, overflow_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
    return host_overflow_value;
}
