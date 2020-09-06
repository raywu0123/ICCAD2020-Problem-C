#ifndef ICCAD2020_CONTAINERS_H
#define ICCAD2020_CONTAINERS_H

#include "constants.h"
#include "memory_manager.h"


template <class T>
struct ResourceCollector {
    std::vector<const std::vector<T>*> vecs;
    unsigned int size = 0; // accumulator of total number of elements
    T *device_ptr, *pinned_ptr;

    explicit ResourceCollector(unsigned int num = 0) {
        vecs.reserve(num);
    }

    unsigned int push(const std::vector<T>& in_vec) {
        auto ret = size;
        vecs.push_back(&in_vec);
        size += in_vec.size();
        return ret;
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


struct ResourceBuffer {

    PinnedMemoryVector<bool*> overflows;
    PinnedMemoryVector<CAPACITY_TYPE> capacities;
    PinnedMemoryVector<const ModuleSpec*> module_specs;
    PinnedMemoryVector<unsigned int> sdf_offsets;
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
    void set(const ResourceBuffer&, cudaStream_t = nullptr);
    void free() const;

    bool** overflows;
    unsigned int* capacities;
    const ModuleSpec** module_specs;
    unsigned int *sdf_offsets, *sdf_num_rows;
    InputData* input_data_schedule;
    Data* output_data_schedule;
    unsigned int num_modules;
};

#endif //ICCAD2020_CONTAINERS_H
