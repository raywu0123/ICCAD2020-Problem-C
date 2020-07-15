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
}

void IndexedWire::store_to_bucket() const {
    auto num_ptrs = first_free_data_ptr_index;
    wire->store_to_bucket(data_ptrs, num_ptrs, capacity);
}

unsigned int ScheduledWire::load(int session_index, const vector<unsigned int>& starting_indices, unsigned int progress_index) {
//        FIXME what if bucket is empty?
    auto* ptr = IndexedWire::alloc(session_index);

    for (unsigned int stimuli_index = 0; stimuli_index < N_STIMULI_PARALLEL; stimuli_index++) {
        if (progress_index >= starting_indices.size() - 1) break;
        unsigned int size = 1; // one for header
        unsigned int end_progress_index = progress_index;
        while (size <= capacity) {
            if (end_progress_index >= starting_indices.size() - 1) break;
            size += starting_indices[end_progress_index + 1] - starting_indices[end_progress_index];
            end_progress_index++;
        }
        if (size > capacity) end_progress_index--;
        Wire::load_from_bucket(
            ptr,
            capacity,
            stimuli_index,
            scheduled_bucket,
            starting_indices[progress_index] - 1, // minus one for header
            starting_indices[end_progress_index]
        );
        progress_index = end_progress_index;
    }
    return progress_index;
}

void ScheduledWire::free() { IndexedWire::free(); vector<Transition>().swap(scheduled_bucket); }

unsigned int ScheduledWire::size() const { return wire->bucket.size(); }
