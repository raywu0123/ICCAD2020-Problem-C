#include <iostream>
#include "data_structures.h"

std::ostream& operator<< (std::ostream& os, const Transition& transition) {
    os << "(" << transition.timestamp << ", " << transition.value << ")";
    return os;
}

void BatchResource::init(const ResourceBuffer& resource_buffer) {
    num_modules = resource_buffer.size();

    cudaMalloc((void**) &overflows, sizeof(bool*) * num_modules);
    cudaMalloc((void**) &module_specs, sizeof(ModuleSpec*) * num_modules);
    cudaMalloc((void**) &sdf_specs, sizeof(SDFSpec*) * num_modules);
    cudaMalloc((void**) &data_schedule, sizeof(Data) * resource_buffer.data_schedule.size());
    cudaMalloc((void**) &data_schedule_offsets, sizeof(unsigned int) * num_modules);

    cudaMemcpy(overflows, resource_buffer.overflows.data(), sizeof(bool*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(sdf_specs, resource_buffer.sdf_specs.data(), sizeof(SDFSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(data_schedule, resource_buffer.data_schedule.data(), sizeof(Data) * resource_buffer.data_schedule.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(data_schedule_offsets, resource_buffer.data_schedule_offsets.data(), sizeof(unsigned int) * num_modules, cudaMemcpyHostToDevice);
}

void BatchResource::free() const {
    cudaFree(overflows);
    cudaFree(module_specs);
    cudaFree(sdf_specs);
    cudaFree(data_schedule);
    cudaFree(data_schedule_offsets);
}
