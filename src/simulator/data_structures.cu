#include <iostream>

#include "constants.h"
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

void BatchResource::init(const ResourceBuffer& resource_buffer) {
    num_modules = resource_buffer.size;

    cudaMalloc((void**) &overflows, sizeof(bool*) * num_modules);
    cudaMalloc((void**) &capacities, sizeof(unsigned int) * num_modules);
    cudaMalloc((void**) &module_specs, sizeof(ModuleSpec*) * num_modules);
    cudaMalloc((void**) &sdf_specs, sizeof(SDFSpec*) * num_modules);
    cudaMalloc((void**) &data_schedule, sizeof(Data) * resource_buffer.data_schedule.size());

    cudaMemcpy(overflows, resource_buffer.overflows.data(), sizeof(bool*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(capacities, resource_buffer.capacities.data(), sizeof(unsigned int) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(sdf_specs, resource_buffer.sdf_specs.data(), sizeof(SDFSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(data_schedule, resource_buffer.data_schedule.data(), sizeof(Data) * resource_buffer.data_schedule.size(), cudaMemcpyHostToDevice);
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
