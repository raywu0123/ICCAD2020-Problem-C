#ifndef ICCAD2020_CONTAINERS_H
#define ICCAD2020_CONTAINERS_H

#include "constants.h"
#include "memory_manager.h"

struct SDFCollector {
    std::vector<SDFPath> paths;
    SDFPath *device_sdf, *pinned_sdf;

    unsigned int push(const std::vector<SDFPath>& cell_paths);
    SDFPath* get();
    void free() const;
};


struct ResourceBuffer {

    PinnedMemoryVector<bool*> overflows;
    PinnedMemoryVector<CAPACITY_TYPE> capacities;
    PinnedMemoryVector<const ModuleSpec*> module_specs;
    PinnedMemoryVector<unsigned int> sdf_offsets;
    PinnedMemoryVector<unsigned int> sdf_num_rows;
    PinnedMemoryVector<Data> data_schedule;

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
    Data* data_schedule;
    unsigned int num_modules;
};

#endif //ICCAD2020_CONTAINERS_H
