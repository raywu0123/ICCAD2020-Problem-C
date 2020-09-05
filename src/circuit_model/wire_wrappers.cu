#include <iostream>

#include "cell.h"
#include "utils.h"

using namespace std;

Data IndexedWire::alloc(int session_index, cudaStream_t stream) {
    if (session_index != previous_session_index) {
        first_free_data_ptr_index = 0;
        previous_session_index = session_index;
    }
    unsigned int size = sizeof(Transition) * static_cast<unsigned int>(capacity) * N_STIMULI_PARALLEL;

    if (first_free_data_ptr_index >= data_list.size())
        data_list.emplace_back(MemoryManager::alloc(size), MemoryManager::alloc(sizeof(unsigned int)));

    if (first_free_data_ptr_index >= data_list.size())
        throw std::runtime_error("Invalid access to data_ptrs");

    auto& data = data_list[first_free_data_ptr_index];
    cudaMemsetAsync(data.transitions, 0, size, stream);
    cudaMemsetAsync(data.size, 0, sizeof(unsigned int), stream);
    first_free_data_ptr_index++;
    return data;
}

Data IndexedWire::load(int session_index, cudaStream_t stream) { return alloc(session_index, stream); }

void IndexedWire::free() {
    for (const auto& data : data_list) {
        MemoryManager::free(data.transitions, sizeof(Transition) * static_cast<unsigned int>(capacity) * N_STIMULI_PARALLEL);
        MemoryManager::free(data.size, sizeof(unsigned int));
    }
    data_list.clear();
    first_free_data_ptr_index = 0;
}

void IndexedWire::store_to_bucket(cudaStream_t stream) const {
    const auto& num_data = first_free_data_ptr_index;
    wire->store_to_bucket(data_list, num_data, stream);
}

void IndexedWire::handle_overflow() {
    free();
    first_free_data_ptr_index = 0;
}

void IndexedWire::finish() {
    free();
    wire->bucket.transitions.shrink_to_fit();
}

Data ScheduledWire::load(int session_index, cudaStream_t stream) {
    const auto& data = IndexedWire::alloc(session_index, stream);
    if (session_index > checkpoint.first) checkpoint = make_pair(session_index, bucket_idx);

    auto start_index = bucket_index_schedule[bucket_idx];
    const auto& end_index = bucket_index_schedule[bucket_idx + 1];
    if (start_index != 0) start_index--;
    wire->load_from_bucket(data.transitions, start_index, end_index, stream);
    bucket_idx++;
    return data;
}

void ScheduledWire::free() { IndexedWire::free(); }

unsigned int ScheduledWire::size() const { return wire->bucket.size(); }

bool ScheduledWire::finished() const {
    return bucket_idx + 1 >= bucket_index_schedule.size();
}

void ScheduledWire::push_back_schedule_index(unsigned int i) {
    if (i > wire->bucket.size())
        throw std::runtime_error("Schedule index out of range.");
    if (not bucket_index_schedule.empty() and i < bucket_index_schedule.back())
        throw std::runtime_error("Schedule index in incorrect order.");
    bucket_index_schedule.push_back(i);
}

void ScheduledWire::handle_overflow() {
    first_free_data_ptr_index = 0;
    bucket_idx = checkpoint.second;
}

void ScheduledWire::finish() {
    free();
    wire->bucket.transitions.shrink_to_fit();
    vector<unsigned int>().swap(bucket_index_schedule);
}

