#include "data_structures.h"

BatchResource::BatchResource(const ResourceBuffer& resource_buffer) {
    num_modules = resource_buffer.size();

    cudaMalloc((void**) &module_specs, sizeof(ModuleSpec*) * num_modules);
    cudaMalloc((void**) &sdf_specs, sizeof(SDFSpec*) * num_modules);
    cudaMalloc((void**) &data_schedule, sizeof(Transition*) * resource_buffer.data_schedule.size());
    cudaMalloc((void**) &data_schedule_offsets, sizeof(unsigned int) * num_modules);
    cudaMalloc((void**) &capacities, sizeof(unsigned int) * resource_buffer.capacities.size());

    cudaMemcpy(module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(sdf_specs, resource_buffer.sdf_specs.data(), sizeof(SDFSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(data_schedule, resource_buffer.data_schedule.data(), sizeof(Transition*) * resource_buffer.data_schedule.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(data_schedule_offsets, resource_buffer.data_schedule_offsets.data(), sizeof(unsigned int) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(capacities, resource_buffer.capacities.data(), sizeof(unsigned int) * resource_buffer.capacities.size(), cudaMemcpyHostToDevice);
}

BatchResource::~BatchResource() {
    cudaFree(module_specs);
    cudaFree(sdf_specs);
    cudaFree(data_schedule);
    cudaFree(data_schedule_offsets);
    cudaFree(capacities);
}
