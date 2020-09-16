#ifndef ICCAD2020_CONTAINERS_H
#define ICCAD2020_CONTAINERS_H

#include <iostream>
#include <utility>

#include "constants.h"
#include "memory_manager.h"


template <class T, class K>
struct ResourceCollector {
    std::vector<const std::vector<T>*> vecs;
    std::unordered_map<const K*, unsigned int> offsets;

    unsigned int size = 0; // accumulator of total number of elements
    T *device_ptr, *pinned_ptr;

    explicit ResourceCollector(unsigned int num = 0) {
        vecs.reserve(num);
    }

    unsigned int push(const std::vector<T>& in_vec, const K* key) {
        auto it = offsets.find(key);
        if (it == offsets.end()) {
            auto ret = size;
            vecs.push_back(&in_vec);
            size += in_vec.size();
            offsets[key] = ret;
            return ret;
        } else return it->second;
    }

    T* get() {
        cudaMallocHost((void**) &pinned_ptr, sizeof(T) * size);
        unsigned int offset = 0;
        for (auto* vec_ptr : vecs) {
            memcpy(pinned_ptr + offset, vec_ptr->data(), sizeof(T) * vec_ptr->size());
            offset += vec_ptr->size();
        }
        std::vector<const std::vector<T>*>().swap(vecs);

        cudaMalloc((void**) &device_ptr, sizeof(T) * size);
        cudaMemcpyAsync(device_ptr, pinned_ptr, sizeof(T) * size, cudaMemcpyHostToDevice);
        return device_ptr;
    }

    void free() const {
        cudaFreeHost(pinned_ptr);
        cudaFree(device_ptr);
    }
};

template<class T>
struct OutputCollector {

    OutputCollector() = default;
    explicit OutputCollector(unsigned int reserve_size) { reserve(reserve_size); }

    void reserve(unsigned int reserve_size) {
        if (current_alloced_size >= reserve_size) return;
        free();
        current_alloced_size = reserve_size;
        cudaMalloc((void**) &device_ptr, sizeof(T) * current_alloced_size);
        cudaMallocHost((void**) &host_ptr, sizeof(T) * current_alloced_size);
    }

    unsigned int push(unsigned int size) {
        unsigned int ret = size_accumulator;
        size_accumulator += size;
        return ret;
    }

    void reset() {
        size_accumulator = 0;
    }

    T* get_device(cudaStream_t stream) {
        if (size_accumulator > current_alloced_size) {
            cudaFree(device_ptr); cudaFreeHost(host_ptr);
            current_alloced_size = size_accumulator * 2;
            cudaMalloc((void**) &device_ptr, sizeof(T) * current_alloced_size);
            cudaMallocHost((void**) &host_ptr, sizeof(T) * current_alloced_size);
        }
        cudaMemsetAsync(device_ptr, 0, sizeof(T) * size_accumulator, stream);
        return device_ptr;
    }

    T* get_host(cudaStream_t stream) {
        cudaMemcpyAsync(host_ptr, device_ptr, sizeof(T) * size_accumulator, cudaMemcpyDeviceToHost, stream);
        return host_ptr;
    }

    void free() {
        cudaFreeHost(host_ptr); host_ptr = nullptr;
        cudaFree(device_ptr); device_ptr = nullptr;
        size_accumulator = 0;
    }

    T *device_ptr = nullptr, *host_ptr = nullptr;
    unsigned int current_alloced_size = 0;
    unsigned int size_accumulator = 0;
};


struct ResourceBuffer {

    PinnedMemoryVector<bool*> overflows;
    PinnedMemoryVector<CAPACITY_TYPE> capacities;
    PinnedMemoryVector<const ModuleSpec*> module_specs;
    PinnedMemoryVector<unsigned int> sdf_offsets, s_timestamp_offsets, s_delay_info_offsets, s_value_offsets, s_length_offsets;
    PinnedMemoryVector<unsigned int> sdf_num_rows;
    PinnedMemoryVector<InputData> input_data_schedule;
    PinnedMemoryVector<Data> output_data_schedule;

    ResourceBuffer();
    void finish_module();
    void clear();
    unsigned int size = 0;
};


struct BatchResource {
    void init(cudaStream_t = nullptr);
    void set(const ResourceBuffer&, cudaStream_t);
    void free() const;

    bool** overflows;
    unsigned int* capacities;
    const ModuleSpec** module_specs;
    unsigned int *sdf_offsets, *s_timestamp_offsets, *s_delay_info_offsets, *s_value_offsets, *s_length_offsets, *sdf_num_rows;
    InputData* input_data_schedule;
    Data* output_data_schedule;
    unsigned int num_modules;
};

#endif //ICCAD2020_CONTAINERS_H
