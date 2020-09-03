#ifndef ICCAD2020_CONTAINERS_H
#define ICCAD2020_CONTAINERS_H

#include "constants.h"
#include "memory_manager.h"

struct ResourceBuffer {

    PinnedMemoryVector<bool*> overflows;
    PinnedMemoryVector<CAPACITY_TYPE> capacities;
    PinnedMemoryVector<const ModuleSpec*> module_specs;
    PinnedMemoryVector<const SDFSpec*> sdf_specs;
    PinnedMemoryVector<Data> data_schedule;

    ResourceBuffer ();
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
    const SDFSpec** sdf_specs;
    Data* data_schedule;
    unsigned int num_modules;
};

#endif //ICCAD2020_CONTAINERS_H
