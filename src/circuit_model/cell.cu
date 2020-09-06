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
) : module_spec(module_spec), name(std::move(name)), declare(declare)
{
    num_args = declare->num_input + declare->num_output;
    build_wire_map(pin_specs);
}

void Cell::build_wire_map(const WireMap<Wire>& pin_specs) {
    if (num_args > MAX_NUM_MODULE_ARGS) {
        throw runtime_error("Too many module args (" + to_string(num_args) + ")\n");
    }
    input_wires.resize(declare->num_input);
    for (NUM_ARG_TYPE arg = 0; arg < declare->num_input; ++arg) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        auto* scheduled_wire = new InputWire(wire_ptr);
        input_wires[arg] = scheduled_wire;
    }

    output_wires.resize(declare->num_output);
    for (NUM_ARG_TYPE arg = declare->num_input; arg < num_args; ++arg) {
        auto* wire_ptr = pin_specs.get(arg);
        if (wire_ptr == nullptr) continue;
        wire_ptr->set_drived();
        auto* indexed_wire = new OutputWire(wire_ptr, output_capacity);
        output_wires[arg - declare->num_input] = indexed_wire;
    }
}

void Cell::set_stream(cudaStream_t s) { stream = s; }

void Cell::init(ResourceCollector<SDFPath>& sdf_collector, ResourceCollector<Transition>& input_data_collector) {
    overflow_ptr = static_cast<bool*>(MemoryManager::alloc(sizeof(bool)));
    host_overflow_ptr = static_cast<bool*>(MemoryManager::alloc_host(sizeof(bool)));
    Cell::build_bucket_index_schedule(
            input_wires,
            (INITIAL_CAPACITY * N_STIMULI_PARALLEL) - 1
    );
    for (const auto& input_wire : input_wires) {
        if (input_wire != nullptr) input_wire->wire->to_device(input_data_collector);
    }
    sdf_offset = sdf_collector.push(sdf_paths);
}

void Cell::free() {
    MemoryManager::free(overflow_ptr, sizeof(bool));
    MemoryManager::free_host(host_overflow_ptr, sizeof(bool));

    vector<SDFPath>().swap(sdf_paths);
    for (auto& input_wire : input_wires) {
        if (input_wire != nullptr) input_wire->finish();
    }
    for (auto& output_wire : output_wires) {
        if (output_wire != nullptr) output_wire->finish();
    }
}

void Cell::prepare_resource(int session_id, ResourceBuffer& resource_buffer) {
    cudaMemsetAsync(overflow_ptr, 0, sizeof(bool), stream);
    *host_overflow_ptr = false;

    resource_buffer.overflows.push_back(overflow_ptr);
    resource_buffer.capacities.push_back(output_capacity);
    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.sdf_offsets.push_back(sdf_offset);
    resource_buffer.sdf_num_rows.push_back(sdf_paths.size());

    for (auto* input_wire : input_wires) {
        if (input_wire == nullptr) resource_buffer.input_data_schedule.push_back(InputData{});
        else resource_buffer.input_data_schedule.push_back(input_wire->load(session_id, stream));
    }
    for (auto* output_wire : output_wires) {
        if (output_wire == nullptr) resource_buffer.output_data_schedule.push_back(Data{});
        else resource_buffer.output_data_schedule.push_back(output_wire->load(session_id, stream));
    }
    resource_buffer.finish_module();
}

void Cell::gather_results_pre() {
    for (const auto& output_wire : output_wires) {
        if (output_wire != nullptr) output_wire->gather_result_pre();
    }
}

void Cell::gather_results_async() {
    for (const auto& output_wire : output_wires) {
        if (output_wire != nullptr) output_wire->gather_result_async(stream);
    }
}

void Cell::gather_results_finalize() {
    for (const auto& output_wire : output_wires) {
        if (output_wire != nullptr) output_wire->finalize_result();
    }
}

void Cell::gather_overflow_async() {
    cudaMemcpyAsync(host_overflow_ptr, overflow_ptr, sizeof(bool), cudaMemcpyDeviceToHost, stream);
}

bool Cell::handle_overflow() {
    if (*host_overflow_ptr) {
        for (auto& input_wire : input_wires) {
            if (input_wire != nullptr) input_wire->handle_overflow();
        }
        for (auto& output_wire : output_wires) {
            if (output_wire != nullptr) output_wire->handle_overflow();
        }
        output_capacity *= 2;
    }
    return *host_overflow_ptr;
}

bool Cell::finished() const {
    for (const auto& input_wire : input_wires) {
        if (input_wire != nullptr and not input_wire->finished()) return false;
    }
    return true;
}

void Cell::build_bucket_index_schedule(vector<InputWire*>& wires, unsigned int size) {
    NUM_ARG_TYPE num_finished = 0, num_inputs = wires.size();

    vector<unsigned int> starting_indices; starting_indices.resize(num_inputs);
    vector<bool> finished; finished.resize(num_inputs);
    for (int i_wire = 0; i_wire < num_inputs; i_wire++) {
        const auto& wire = wires[i_wire];
        if (wire == nullptr or size >= wire->wire->bucket.size()) {
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
