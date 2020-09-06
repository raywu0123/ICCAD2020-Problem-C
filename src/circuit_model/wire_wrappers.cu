#include <iostream>

#include "cell.h"
#include "utils.h"

using namespace std;

Data& OutputWire::alloc(int session_index, cudaStream_t stream) {
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

Data OutputWire::load(int session_index, cudaStream_t stream) { return alloc(session_index, stream); }

void OutputWire::free() {
    for (const auto& data : data_list) {
        MemoryManager::free(data.transitions, sizeof(Transition) * static_cast<unsigned int>(capacity) * N_STIMULI_PARALLEL);
        MemoryManager::free(data.size, sizeof(unsigned int));
    }
    data_list.clear();
    first_free_data_ptr_index = 0;
}

void OutputWire::gather_result_pre() {
    const auto& num_data = first_free_data_ptr_index;
    auto data_size = capacity * N_STIMULI_PARALLEL;
    for (int i = 0; i < num_data; ++i) {
        host_data_storage.emplace_back(
            static_cast<Transition*>(MemoryManager::alloc_host( sizeof(Transition) * data_size)),
            static_cast<unsigned int*>(MemoryManager::alloc_host(sizeof(unsigned int)))
        );
    }
}

void OutputWire::gather_result_async(cudaStream_t stream) {
    const auto& num_data = first_free_data_ptr_index;
    auto data_size = capacity * N_STIMULI_PARALLEL;

    const auto& direction = cudaMemcpyDeviceToHost;
    for (int i = 0; i < num_data; ++i) {
        cudaMemcpyAsync(
            host_data_storage[i].transitions, data_list[i].transitions,
            sizeof(Transition) * data_size, direction, stream
        );
        cudaMemcpyAsync(
            host_data_storage[i].size, data_list[i].size,
            sizeof(unsigned int), direction, stream
        );
    }
}

void OutputWire::finalize_result() {
    wire->store_to_bucket(host_data_storage);
    auto data_size = capacity * N_STIMULI_PARALLEL;

    for (auto& data: host_data_storage) {
        MemoryManager::free_host(data.transitions, sizeof(Transition) * data_size);
        MemoryManager::free_host(data.size, sizeof(unsigned int));
    }
    host_data_storage.clear();
}

void OutputWire::handle_overflow() {
    free();
    first_free_data_ptr_index = 0;
}

void OutputWire::finish() {
    free();
    wire->bucket.transitions.shrink_to_fit();
}

InputData& InputWire::alloc(int session_index, cudaStream_t stream) {
    if (session_index != previous_session_index) {
        first_free_data_ptr_index = 0;
        previous_session_index = session_index;
    }

    if (first_free_data_ptr_index >= data_list.size())
        data_list.push_back({});

    if (first_free_data_ptr_index >= data_list.size())
        throw std::runtime_error("Invalid access to data_ptrs");

    auto& data = data_list[first_free_data_ptr_index];
    first_free_data_ptr_index++;
    return data;
}

InputData InputWire::load(int session_index, cudaStream_t stream) {
    auto& data = alloc(session_index, stream);
    if (session_index > checkpoint.first) checkpoint = make_pair(session_index, bucket_idx);

    auto start_index = bucket_index_schedule[bucket_idx];
    const auto& end_index = bucket_index_schedule[bucket_idx + 1];
    if (start_index != 0) start_index--;
    bucket_idx++;

    data.size = end_index - start_index;
    data.transitions = wire->device_ptr + start_index;
    return data;
}

void InputWire::free() {
    data_list.clear();
    wire->free_device();
    first_free_data_ptr_index = 0;
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
    free();
    wire->bucket.transitions.shrink_to_fit();
    vector<unsigned int>().swap(bucket_index_schedule);
}

