#ifndef ICCAD2020_BUILTIN_GATES_H
#define ICCAD2020_BUILTIN_GATES_H

#include "constants.h"

__host__ __device__ char and_logic(
    Transition** data, unsigned int num_inputs, unsigned int* indices
) {
    bool is_all_one = true, has_zero = false;
    for (int i = 0; i < num_inputs; i++) {
        const auto& value = data[i][indices[i]].value;
        if (value == '0') { has_zero = true; }
        if (value != '1') { is_all_one = false; }
    }
    if (has_zero) return '0';
    if (is_all_one) return '1';
    return 'x';
}

__host__ __device__ void merge_sort_algorithm(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    unsigned int* capacities,
    char* table,
    unsigned int num_inputs, unsigned int num_outputs,
    LogicFn logic_fn
) {
    unsigned int num_finished = 0, i_output = 0;
    auto* indices = new unsigned int[num_inputs];
    for (int i = 0; i < num_inputs; i++) indices[i] = 0;
    while (num_finished < num_inputs) {
        unsigned min_i_input;
        Timestamp min_timestamp = LONG_LONG_MAX;
        for (int i = 0; i < num_inputs; i++) {
            if (indices[i] + 1 >= capacities[i]) continue;

            const auto& transition = data[i][indices[i] + 1];
            if (transition.timestamp < min_timestamp) {
                min_timestamp = transition.timestamp;
                min_i_input = i;
            }
        }
        indices[min_i_input]++;
        data[num_inputs][i_output].timestamp = data[min_i_input][indices[min_i_input]].timestamp;
        data[num_inputs][i_output].value = logic_fn(data, num_inputs, indices);
        i_output++;
        if (i_output >= capacities[num_inputs]) break; // TODO handle overflow
        if (indices[min_i_input] >= capacities[min_i_input] - 1) num_finished++;
    }
    delete[] indices;
}

// Gates compute results on single stimuli
__host__ __device__ void and_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    unsigned int* capacities,
    char* table,
    unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, num_inputs, num_outputs, and_logic);
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
