#include <iostream>

#include "cell.h"
#include "utils.h"

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

ScheduledWire::ScheduledWire(Wire *wire): IndexedWire(wire) {
    cudaErrorCheck(cudaMalloc((void**) &progress_update_ptr, sizeof(unsigned int)));
}

Transition* ScheduledWire::load(int session_index) {
    auto* ptr = IndexedWire::load(session_index);
    unsigned int end_index = min(wire->bucket.size(), bucket_idx - 1 + capacity * N_STIMULI_PARALLEL);
    Wire::load_from_bucket(ptr, wire->bucket.transitions, bucket_idx - 1, end_index);
    return ptr;
}

void ScheduledWire::update_progress() {
    unsigned int host_update_progress = 0;
    cudaMemcpy(&host_update_progress, progress_update_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    bucket_idx += host_update_progress;
}

void ScheduledWire::free() { IndexedWire::free(); }

unsigned int ScheduledWire::size() const { return wire->bucket.size(); }

bool ScheduledWire::finished() const {
    return bucket_idx >= wire->bucket.size();
}

