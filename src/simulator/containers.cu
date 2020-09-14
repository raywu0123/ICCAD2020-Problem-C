#include <cassert>

#include "containers.h"

using namespace std;

void BatchResource::init(cudaStream_t) {
    cudaMalloc((void**) &overflows, sizeof(bool*) * N_CELL_PARALLEL);
    cudaMalloc((void**) &capacities, sizeof(CAPACITY_TYPE) * N_CELL_PARALLEL);
    cudaMalloc((void**) &module_specs, sizeof(ModuleSpec*) * N_CELL_PARALLEL);
    cudaMalloc((void**) &sdf_offsets, sizeof(unsigned int) * N_CELL_PARALLEL);
    cudaMalloc((void**) &s_timestamp_offsets, sizeof(unsigned int) * N_CELL_PARALLEL);
    cudaMalloc((void**) &s_delay_info_offsets, sizeof(unsigned int) * N_CELL_PARALLEL);
    cudaMalloc((void**) &s_value_offsets, sizeof(unsigned int) * N_CELL_PARALLEL);
    cudaMalloc((void**) &s_length_offsets, sizeof(unsigned int) * N_CELL_PARALLEL);

    cudaMalloc((void**) &sdf_num_rows, sizeof(unsigned int) * N_CELL_PARALLEL);
    cudaMalloc((void**) &input_data_schedule, sizeof(InputData) * N_CELL_PARALLEL * MAX_NUM_MODULE_INPUT);
    cudaMalloc((void**) &output_data_schedule, sizeof(Data) * N_CELL_PARALLEL * MAX_NUM_MODULE_OUTPUT);
}

void BatchResource::set(const ResourceBuffer& resource_buffer, cudaStream_t stream) {
    assert(resource_buffer.input_data_schedule.size() <= N_CELL_PARALLEL * MAX_NUM_MODULE_ARGS);
    assert(resource_buffer.output_data_schedule.size() <= N_CELL_PARALLEL * MAX_NUM_MODULE_ARGS);
    num_modules = resource_buffer.size;

    auto direction = cudaMemcpyHostToDevice;
    cudaMemcpyAsync(overflows, resource_buffer.overflows.data(), sizeof(bool*) * num_modules, direction, stream);
    cudaMemcpyAsync(capacities, resource_buffer.capacities.data(), sizeof(CAPACITY_TYPE) * num_modules, direction, stream);
    cudaMemcpyAsync(module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, direction, stream);
    cudaMemcpyAsync(sdf_offsets, resource_buffer.sdf_offsets.data(), sizeof(unsigned int) * num_modules, direction, stream);
    cudaMemcpyAsync(s_timestamp_offsets, resource_buffer.s_timestamp_offsets.data(), sizeof(unsigned int) * num_modules, direction, stream);
    cudaMemcpyAsync(s_delay_info_offsets, resource_buffer.s_delay_info_offsets.data(), sizeof(unsigned int) * num_modules, direction, stream);
    cudaMemcpyAsync(s_value_offsets, resource_buffer.s_value_offsets.data(), sizeof(unsigned int) * num_modules, direction, stream);
    cudaMemcpyAsync(s_length_offsets, resource_buffer.s_length_offsets.data(), sizeof(unsigned int) * num_modules, direction, stream);

    cudaMemcpyAsync(sdf_num_rows, resource_buffer.sdf_num_rows.data(), sizeof(unsigned int) * num_modules, direction, stream);
    cudaMemcpyAsync(input_data_schedule, resource_buffer.input_data_schedule.data(), sizeof(InputData) * resource_buffer.input_data_schedule.size(), direction, stream);
    cudaMemcpyAsync(output_data_schedule, resource_buffer.output_data_schedule.data(), sizeof(Data) * resource_buffer.output_data_schedule.size(), direction, stream);
}

void BatchResource::free() const {
    cudaFree(overflows);
    cudaFree(capacities);
    cudaFree(module_specs);
    cudaFree(sdf_offsets);
    cudaFree(s_timestamp_offsets); cudaFree(s_delay_info_offsets); cudaFree(s_value_offsets); cudaFree(s_length_offsets);
    cudaFree(sdf_num_rows);
    cudaFree(input_data_schedule); cudaFree(output_data_schedule);
}

ResourceBuffer::ResourceBuffer() {
    overflows.reserve(N_CELL_PARALLEL);
    capacities.reserve(N_CELL_PARALLEL);
    module_specs.reserve(N_CELL_PARALLEL);
    sdf_offsets.reserve(N_CELL_PARALLEL);
    s_timestamp_offsets.reserve(N_CELL_PARALLEL); s_delay_info_offsets.reserve(N_CELL_PARALLEL); s_value_offsets.reserve(N_CELL_PARALLEL); s_length_offsets.reserve(N_CELL_PARALLEL);
    sdf_num_rows.reserve(N_CELL_PARALLEL);
    input_data_schedule.reserve(N_CELL_PARALLEL * MAX_NUM_MODULE_INPUT);
    output_data_schedule.reserve(N_CELL_PARALLEL * MAX_NUM_MODULE_OUTPUT);
}

void ResourceBuffer::finish_module() {
    size++;
    input_data_schedule.resize(size * MAX_NUM_MODULE_INPUT);
    output_data_schedule.resize(size * MAX_NUM_MODULE_OUTPUT);
}

void ResourceBuffer::clear() {
    overflows.clear();
    capacities.clear();
    module_specs.clear();
    sdf_offsets.clear(); s_timestamp_offsets.clear(); s_delay_info_offsets.clear(); s_value_offsets.clear(); s_length_offsets.clear();
    sdf_num_rows.clear();
    input_data_schedule.clear();
    output_data_schedule.clear();
    size = 0;
}

