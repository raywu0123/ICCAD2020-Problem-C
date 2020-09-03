#include <cassert>

#include "containers.h"


void BatchResource::init(cudaStream_t) {
    cudaMalloc((void**) &overflows, sizeof(bool*) * N_CELL_PARALLEL);
    cudaMalloc((void**) &capacities, sizeof(CAPACITY_TYPE) * N_CELL_PARALLEL);
    cudaMalloc((void**) &module_specs, sizeof(ModuleSpec*) * N_CELL_PARALLEL);
    cudaMalloc((void**) &sdf_specs, sizeof(SDFSpec*) * N_CELL_PARALLEL);
    cudaMalloc((void**) &data_schedule, sizeof(Data) * N_CELL_PARALLEL * MAX_NUM_MODULE_ARGS);
}

void BatchResource::set(const ResourceBuffer& resource_buffer, cudaStream_t stream) {
    assert(resource_buffer.data_schedule.size() <= N_CELL_PARALLEL * MAX_NUM_MODULE_ARGS);
    num_modules = resource_buffer.size;

    auto direction = cudaMemcpyHostToDevice;
    cudaMemcpyAsync(overflows, resource_buffer.overflows.data(), sizeof(bool*) * num_modules, direction);
    cudaMemcpyAsync(capacities, resource_buffer.capacities.data(), sizeof(CAPACITY_TYPE) * num_modules, direction);
    cudaMemcpyAsync(module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, direction);
    cudaMemcpyAsync(sdf_specs, resource_buffer.sdf_specs.data(), sizeof(SDFSpec*) * num_modules, direction);
    cudaMemcpyAsync(data_schedule, resource_buffer.data_schedule.data(), sizeof(Data) * resource_buffer.data_schedule.size(), direction);
}

void BatchResource::free() const {
    cudaFree(overflows);
    cudaFree(capacities);
    cudaFree(module_specs);
    cudaFree(sdf_specs);
    cudaFree(data_schedule);
}

ResourceBuffer::ResourceBuffer() {
    overflows.reserve(N_CELL_PARALLEL);
    capacities.reserve(N_CELL_PARALLEL);
    module_specs.reserve(N_CELL_PARALLEL);
    sdf_specs.reserve(N_CELL_PARALLEL);
    data_schedule.reserve(N_CELL_PARALLEL * MAX_NUM_MODULE_ARGS);
}

void ResourceBuffer::finish_module() {
    size++;
    data_schedule.resize(size * MAX_NUM_MODULE_ARGS);
}

void ResourceBuffer::clear() {
    overflows.clear();
    capacities.clear();
    module_specs.clear();
    sdf_specs.clear();
    data_schedule.clear();
    size = 0;
}
