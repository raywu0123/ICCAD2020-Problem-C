#include "containers.h"


void BatchResource::init(const ResourceBuffer& resource_buffer, const cudaStream_t& stream) {
    num_modules = resource_buffer.size;

    cudaMalloc((void**) &overflows, sizeof(bool*) * num_modules);
    cudaMalloc((void**) &capacities, sizeof(unsigned int) * num_modules);
    cudaMalloc((void**) &module_specs, sizeof(ModuleSpec*) * num_modules);
    cudaMalloc((void**) &sdf_specs, sizeof(SDFSpec*) * num_modules);
    cudaMalloc((void**) &data_schedule, sizeof(Data) * resource_buffer.data_schedule.size());

    auto direction = cudaMemcpyHostToDevice;
    cudaMemcpyAsync(overflows, resource_buffer.overflows.data(), sizeof(bool*) * num_modules, direction, stream);
    cudaMemcpyAsync(capacities, resource_buffer.capacities.data(), sizeof(unsigned int) * num_modules, direction, stream);
    cudaMemcpyAsync(module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, direction, stream);
    cudaMemcpyAsync(sdf_specs, resource_buffer.sdf_specs.data(), sizeof(SDFSpec*) * num_modules, direction, stream);
    cudaMemcpyAsync(data_schedule, resource_buffer.data_schedule.data(), sizeof(Data) * resource_buffer.data_schedule.size(), direction, stream);
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
