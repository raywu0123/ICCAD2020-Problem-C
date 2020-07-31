#ifndef ICCAD2020_DEVICE_FUNCTIONS_H
#define ICCAD2020_DEVICE_FUNCTIONS_H

#include "circuit_model/circuit.h"

__host__ __device__ int lookup_delay(
        unsigned int, unsigned int, char, char,
        const SDFSpec*
);

__host__ __device__ void compute_delay(
        Transition**, unsigned int capacity, DelayInfo*,
        unsigned int, unsigned int,
        const SDFSpec* sdf_spec, unsigned int* lengths, bool verbose = false
);

__device__ __host__ void slice_waveforms(
        Timestamp* s_timestamps, DelayInfo* s_delay_infos, Values* s_values,
        Data* data, unsigned int capacity,
        unsigned int num_wires, bool* overflow_ptr
);

__device__  void simulate_module(
        const ModuleSpec* module_spec,
        const SDFSpec* sdf_spec,
        Data* data, unsigned int capacity,
        bool* overflow_ptr
);

__global__ void simulate_batch(BatchResource);

#endif //ICCAD2020_DEVICE_FUNCTIONS_H
