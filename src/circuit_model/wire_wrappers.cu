#include <iostream>

#include "cell.h"
#include "utils.h"

using namespace std;

Data OutputWire::load(
    OutputCollector<Transition>& output_data_collector, OutputCollector<unsigned int>& output_size_collector
) {
    auto transition_offset = output_data_collector.push(static_cast<unsigned int>(capacity) * N_STIMULI_PARALLEL);
    auto size_offset = output_size_collector.push(1);

    data_list.emplace_back(transition_offset, size_offset);
    return data_list.back();
}

void OutputWire::gather_result(Transition* output_data, unsigned int* sizes) {
    wire->store_to_bucket(data_list, output_data, sizes);
    data_list.clear();
}

void OutputWire::handle_overflow() {
    data_list.clear();
}

void OutputWire::finish() {
    data_list.clear();
    wire->bucket.transitions.shrink_to_fit();
}

InputData& InputWire::alloc(int session_index) {
    if (session_index != previous_session_index) {
        first_free_data_ptr_index = 0;
        previous_session_index = session_index;
    }

    if (first_free_data_ptr_index >= data_list.size())
        data_list.emplace_back();

    if (first_free_data_ptr_index >= data_list.size())
        throw std::runtime_error("Invalid access to data_ptrs");

    auto& data = data_list[first_free_data_ptr_index];
    first_free_data_ptr_index++;
    return data;
}

InputData InputWire::load(int session_index) {
    auto& data = alloc(session_index);
    if (session_index > checkpoint.first) checkpoint = make_pair(session_index, bucket_idx);

    auto start_index = bucket_index_schedule[bucket_idx];
    const auto& end_index = bucket_index_schedule[bucket_idx + 1];
    if (start_index != 0) start_index--;
    bucket_idx++;

    data.size = end_index - start_index;
    data.offset = wire->offset + start_index;
    return data;
}

unsigned int InputWire::size() const { return wire->bucket.size(); }

bool InputWire::finished() const {
    return bucket_idx + 1 >= bucket_index_schedule.size();
}

void InputWire::push_back_schedule_index(unsigned int i) {
    if (i > wire->bucket.size())
        throw std::runtime_error("Schedule index out of range.");
    if (not bucket_index_schedule.empty() and i < bucket_index_schedule.back())
        throw std::runtime_error("Schedule index in incorrect order.");
    bucket_index_schedule.push_back(i);
}

void InputWire::handle_overflow() {
    first_free_data_ptr_index = 0;
    bucket_idx = checkpoint.second;
}

void InputWire::finish() {
    data_list.clear();
    first_free_data_ptr_index = 0;
    vector<unsigned int>().swap(bucket_index_schedule);
}

