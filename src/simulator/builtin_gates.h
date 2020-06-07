#ifndef ICCAD2020_BUILTIN_GATES_H
#define ICCAD2020_BUILTIN_GATES_H

#include "data_structures.h"

__host__ __device__ char and_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
);
__host__ __device__ char or_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
);
__host__ __device__ char xor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
);
__host__ __device__ char nand_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
);
__host__ __device__ char nor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
);
__host__ __device__ char xnor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
);
__host__ __device__ char not_logic(char value);
__host__ __device__ char buf_logic(char value);

__host__ __device__ void merge_sort_algorithm(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    char* table, unsigned int table_row_num,
    unsigned int num_inputs,
    LogicFn logic_fn,
    bool* overflow
);
__host__ __device__ void single_input_algorithm(
    Transition** data, const unsigned int* capacities, char(*logic_fn)(char), bool* overflow
);

// Gates compute results on single stimuli
__host__ __device__ void and_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    char* table, unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs,
    bool* overflow
);
__host__ __device__ void or_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    char* table, unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs,
    bool* overflow
);
__host__ __device__ void xor_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs,
        bool* overflow
);
__host__ __device__ void nand_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs,
        bool* overflow
);
__host__ __device__ void nor_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs,
        bool* overflow
);
__host__ __device__ void xnor_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs,
        bool* overflow
);
__host__ __device__ void not_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    char* table, unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs,
    bool* overflow
);
__host__ __device__ void buf_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    char* table, unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs,
    bool* overflow
);

__host__ __device__ char primitive_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
);
__host__ __device__ void PrimitiveGate(
    Transition** data,
    const unsigned int* capacities,
    char* table, unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs,
    bool* overflow
);

#endif //ICCAD2020_BUILTIN_GATES_H
