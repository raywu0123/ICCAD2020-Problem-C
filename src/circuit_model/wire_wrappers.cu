#include <iostream>

#include "cell.h"

using namespace std;

Transition* IndexedWire::alloc(int session_index) {
    if (session_index != previous_session_index) {
        first_free_data_ptr_index = 0;
        previous_session_index = session_index;
    }
    unsigned int size = capacity * N_STIMULI_PARALLEL;

    if (first_free_data_ptr_index >= data_ptrs.size())
        data_ptrs.push_back(MemoryManager::alloc(size));

    if (first_free_data_ptr_index >= data_ptrs.size())
        throw std::runtime_error("Invalid access to data_ptrs");

    Transition* data_ptr = data_ptrs[first_free_data_ptr_index];
    cudaMemset(data_ptr, 0, sizeof(Transition) * size);

    first_free_data_ptr_index++;
    return data_ptr;
}

Transition* IndexedWire::load(int session_index) { return alloc(session_index); }

void IndexedWire::free() {
    for (auto* data_ptr : data_ptrs) MemoryManager::free(data_ptr);
    data_ptrs.clear();
    first_free_data_ptr_index = 0;
}

void IndexedWire::store_to_bucket() const {
    auto num_ptrs = first_free_data_ptr_index;
    wire->store_to_bucket(data_ptrs, num_ptrs, capacity);
}

void IndexedWire::handle_overflow() {
    free();
}

Transition* ScheduledWire::load(int session_index) {
//        FIXME what if bucket is empty?
    auto* ptr = IndexedWire::alloc(session_index);
    if (session_index > checkpoint.first) checkpoint = make_pair(session_index, bucket_idx);

    for (unsigned int stimuli_index = 0; stimuli_index < N_STIMULI_PARALLEL; ++stimuli_index) {
        if (finished()) break;
        auto start_index = bucket_index_schedule[bucket_idx];
        const auto& end_index = bucket_index_schedule[bucket_idx + 1];

        if (start_index != 0) start_index--;
        Wire::load_from_bucket(ptr, capacity, stimuli_index, wire->bucket.transitions, start_index, end_index);
        bucket_idx++;
    }
    return ptr;
}

void ScheduledWire::free() { IndexedWire::free(); }

unsigned int ScheduledWire::size() const { return wire->bucket.size(); }

void ScheduledWire::handle_overflow() {
    free();
    bucket_idx = checkpoint.second;
}

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

