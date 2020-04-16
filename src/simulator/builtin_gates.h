#ifndef ICCAD2020_BUILTIN_GATES_H
#define ICCAD2020_BUILTIN_GATES_H

#include "constants.h"

// Gates compute results on single stimuli
__host__ __device__ void and_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    unsigned int* capacities,
    char* table,
    unsigned int num_inputs, unsigned int num_outputs
) {

}

__device__ GateFnPtr and_gate_fn_ptr = and_gate_fn;


__host__ __device__ void PrimitiveGate(
    Transition** data,
    unsigned int* capacities,
    char* table,
    unsigned int num_inputs, unsigned int num_outputs
) {

};

#endif //ICCAD2020_BUILTIN_GATES_H
