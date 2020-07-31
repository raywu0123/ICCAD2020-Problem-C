#include <iostream>

#include "data_structures.h"

using namespace std;

__host__ __device__ EdgeTypes get_edge_type(const Values& v1, const Values& v2) {
    if (v2 == Values::ONE or v1 == Values::ZERO) return EdgeTypes::RISING;
    if (v2 == Values::ZERO or v1 == Values::ONE) return EdgeTypes::FALLING;

    if (v1 == Values::X and v2 == Values::Z) return EdgeTypes::XZ;
    if (v1 == Values::Z and v2 == Values::X) return EdgeTypes::ZX;
    return EdgeTypes::UNDEF;
}

__host__ __device__ char edge_type_to_raw(EdgeTypes e) {
    switch (e) {
        case EdgeTypes::RISING:
            return '+';
        case EdgeTypes::FALLING:
            return '-';
        default:
            return 'x';
    }
}
__host__ __device__ EdgeTypes raw_to_edge_type(char r) {
    switch (r) {
        case '+':
            return EdgeTypes::RISING;
        case '-':
            return EdgeTypes::FALLING;
        default:
            return EdgeTypes::UNDEF;
    }
}

Values raw_to_enum(char v) {
    switch (v) {
        case '0':
            return Values::ZERO;
        case '1':
            return Values::ONE;
        case 'x':
        case 'X':
            return Values::X;
        case 'z':
        case 'Z':
            return Values::Z;
        default:
            return Values::PAD;
    }
}

char enum_to_raw(Values v) {
    switch (v) {
        case Values::ZERO:
            return '0';
        case Values::ONE:
            return '1';
        case Values::X:
            return 'x';
        case Values::Z:
            return 'z';
        default:
            return '_';
    }
}

inline std::ostream& operator<< (std::ostream& os, const Values& v) {
    os << enum_to_raw(v);
    return os;
}

std::ostream& operator<< (std::ostream& os, const Transition& transition) {
    os << "(" << transition.timestamp << ", " << transition.value << ")";
    return os;
}

ResourceBuffer::ResourceBuffer(unsigned int size) {
    module_specs.reserve(size);
    sdf_specs.reserve(size);
    overflow_ptrs.reserve(size);
    data_list.reserve(MAX_NUM_MODULE_ARGS * size);
}

void ResourceBuffer::get_overflows(bool* host_overflows, const cudaStream_t& stream) const {
    for (int i = 0; i < overflow_ptrs.size(); ++i)
        cudaMemcpyAsync(host_overflows + i, overflow_ptrs[i], sizeof(bool), cudaMemcpyDeviceToHost, stream);
}

void BatchResource::init(const ResourceBuffer& resource_buffer, cudaStream_t const &stream) {
    num_modules = resource_buffer.module_specs.size();

    cudaMalloc((void**) &overflows, sizeof(bool*) * num_modules);
    cudaMalloc((void**) &capacities, sizeof(unsigned int) * num_modules);
    cudaMalloc((void**) &module_specs, sizeof(ModuleSpec*) * num_modules);
    cudaMalloc((void**) &sdf_specs, sizeof(SDFSpec*) * num_modules);
    cudaMalloc((void**) &data_list, sizeof(Data) * resource_buffer.data_list.size());

    const auto dir = cudaMemcpyHostToDevice;
    cudaMemcpyAsync(overflows, resource_buffer.overflow_ptrs.data(), sizeof(bool*) * num_modules, dir, stream);
    cudaMemcpyAsync(capacities, resource_buffer.capacities.data(), sizeof(unsigned int) * num_modules, dir, stream);
    cudaMemcpyAsync(module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, dir, stream);
    cudaMemcpyAsync(sdf_specs, resource_buffer.sdf_specs.data(), sizeof(SDFSpec*) * num_modules, dir, stream);
    cudaMemcpyAsync(data_list, resource_buffer.data_list.data(), sizeof(Data) * resource_buffer.data_list.size(), dir, stream);
}

void BatchResource::finish() const {
    cudaFree(data_list);
    cudaFree(module_specs);
    cudaFree(sdf_specs);
    cudaFree(capacities);
    cudaFree(overflows);
}
