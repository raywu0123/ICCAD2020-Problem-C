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
    string name
) : module_spec(module_spec), name(std::move(name))
{
    num_args = declare->num_input + declare->num_output;
    build_wire_map(declare, pin_specs);
}

void Cell::set_paths() {
    vector<char> edge_types;
    vector<unsigned int> input_indices, output_indices;
    vector<int> rising_delays, falling_delays;

    for (const auto& path : sdf_paths) {
        input_indices.push_back(path.in);
        output_indices.push_back(path.out);
        edge_types.push_back(path.edge_type);
        rising_delays.push_back(path.rising_delay);
        falling_delays.push_back(path.falling_delay);
    }

    const auto& num_rows = sdf_paths.size();
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
    const WireMap<Wire>& pin_specs
) {
    if (num_args > MAX_NUM_MODULE_ARGS) {
        throw runtime_error("Too many module args (" + to_string(num_args) + ")\n");
    }
    for (unsigned int arg = 0; arg < declare->num_input; ++arg) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        auto* wrapped_wire = new InputWire(wire_ptr);
        wire_map.set(arg, wrapped_wire); input_wires.push_back(wrapped_wire);
    }
    for (unsigned int arg = declare->num_input; arg < num_args; ++arg) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        wire_ptr->set_drived();
        auto* wrapped_wire = new OutputWire(wire_ptr);
        wire_map.set(arg, wrapped_wire); output_wires.push_back(wrapped_wire);
    }
    for (unsigned int arg = 0; arg < num_args; arg++) {
        if (wire_map.get(arg) == nullptr) cerr << "| WARNING: Arg (" + to_string(arg) + ") not found in wiremap of cell " << name  << endl;
    }
}

void Cell::push_jobs(queue<Job*>& job_queue) {
    init();
    for (int i = 0; i < schedule_size; ++i) {
        vector<JobHandle*> job_handles; job_handles.reserve(num_args);
        for (int arg = 0; arg < num_args; ++arg) {
            auto* wrapped_wire = wire_map.get(arg);
            if (wrapped_wire == nullptr) job_handles.push_back(new JobHandle());
            else job_handles.push_back(wrapped_wire->get_job_handle(i));
        }
        auto* job = new Job(module_spec, sdf_spec, num_args, job_handles);
        job_queue.emplace(job);
    }
}


void Cell::finish() {
    for (auto& wrapped_wire : output_wires) wrapped_wire->finish();

    cudaFree(sdf_spec);
    cudaFree(host_sdf_spec.edge_type);
    cudaFree(host_sdf_spec.input_index); cudaFree(host_sdf_spec.output_index);
    cudaFree(host_sdf_spec.rising_delay); cudaFree(host_sdf_spec.falling_delay);
    for (auto& wrapped_wire : input_wires) wrapped_wire->free();
    for (auto& wrapped_wire : output_wires) wrapped_wire->free();
}

void Cell::init() {
    set_paths();
    schedule_size = Cell::build_bucket_index_schedule(
        input_wires,
        (INITIAL_CAPACITY * N_STIMULI_PARALLEL) / input_wires.size() - 1
    );
    for (auto& wrapped_wire : output_wires) wrapped_wire->set_schedule_size(schedule_size);
}

unsigned int Cell::build_bucket_index_schedule(vector<InputWire*>& wires, unsigned int size) {
    unsigned int num_finished = 0, num_inputs = wires.size();

    vector<unsigned int> starting_indices; starting_indices.resize(num_inputs);
    vector<bool> finished; finished.resize(num_inputs);
    for (int i_wire = 0; i_wire < num_inputs; i_wire++) {
        const auto& wire = wires[i_wire];
        if (size >= wire->wire->bucket.size()) {
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
            unsigned int end_index = starting_indices[i_wire] + size - 1;
            if (end_index >= bucket.size()) continue;
            const auto& end_timestamp = bucket[end_index].timestamp;
            if (end_timestamp < min_end_timestamp) min_end_timestamp = end_timestamp;
        }

        for (int i_wire = 0; i_wire < num_inputs; i_wire++) {
            auto& wire = wires[i_wire];
            const auto& bucket = wire->wire->bucket;
            const auto& bucket_size = bucket.size();
//            If already finished, push_back the last index of bucket
            if (not wire->bucket_index_schedule.empty() and wire->bucket_index_schedule.back() == bucket_size) {
                wire->push_back_schedule_index(bucket_size);
            } else {
//                FIXME will fail if start_index = 0 and timestamp[0] > min_end_timestamp
                auto start_index = bucket[starting_indices[i_wire]].timestamp > min_end_timestamp ? starting_indices[i_wire] - 1 : starting_indices[i_wire];
                auto end_index = find_end_index(bucket, start_index, min_end_timestamp, size);
                auto next_start_index = end_index + 1;
                wire->push_back_schedule_index(next_start_index);
                if (next_start_index + size >= bucket.size() and not finished[i_wire]) {
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
    return wires.front()->bucket_index_schedule.size() - 1;
}

unsigned int Cell::find_end_index(const PinnedMemoryVector<Transition>& bucket, unsigned int start_index, const Timestamp& t, unsigned int capacity) {
//    Binary Search for end_index <= t
    unsigned int low = start_index, high = min(start_index + capacity, (unsigned int) bucket.size()) - 1;
    if (bucket[high].timestamp <= t) return high;
    while (low < high) {
        unsigned mid = (low + high) / 2;
        if (mid == low) break;
        if (bucket[mid].timestamp < t) low = mid;
        else if (bucket[mid].timestamp > t) high = mid;
        else return mid;
    }
    return low;
}
