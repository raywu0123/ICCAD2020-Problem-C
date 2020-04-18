#include "builtin_gates.h"


__host__ __device__ char and_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
) {
    bool is_all_one = true, has_zero = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        if (value == '0') { has_zero = true; }
        if (value != '1') { is_all_one = false; }
    }
    if (has_zero) return '0';
    if (is_all_one) return '1';
    return 'x';
}

__host__ __device__ char or_logic(
        Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
) {
    bool is_all_zero = true, has_one = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        if (value == '1') { has_one = true; }
        if (value != '0') { is_all_zero = false; }
    }
    if (has_one) return '1';
    if (is_all_zero) return '0';
    return 'x';
}
__host__ __device__ char xor_logic(
        Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
) {
    char ret = '0';
    bool has_xz = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        if (value == 'x' or value == 'z') has_xz = true;
        ret = ret == value ? '0' : '1';
    }
    if (has_xz) return 'x';
    return ret;
}
__host__ __device__ char nand_logic(
        Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
) {
    bool is_all_one = true, has_zero = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        if (value == '0') { has_zero = true; }
        if (value != '1') { is_all_one = false; }
    }
    if (has_zero) return '1';
    if (is_all_one) return '0';
    return 'x';
}
__host__ __device__ char nor_logic(
        Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
) {
    bool is_all_zero = true, has_one = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        if (value == '1') { has_one = true; }
        if (value != '0') { is_all_zero = false; }
    }
    if (has_one) return '0';
    if (is_all_zero) return '1';
    return 'x';
}
__host__ __device__ char xnor_logic(
        Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
) {
    char ret = '0';
    bool has_xz = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        if (value == 'x' or value == 'z') has_xz = true;
        ret = ret == value ? '1' : '0';
    }
    if (has_xz) return 'x';
    return ret;
}
__host__ __device__ char not_logic(char value) {
    if (value == '0') return '1';
    if (value == '1') return '0';
    return 'x';
}
__host__ __device__ char buf_logic(char value) {
    if (value == 'x' or value == 'z') return 'x';
    return value;
}

__host__ __device__ void merge_sort_algorithm(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs,
        LogicFn logic_fn
) {
    unsigned int num_finished = 0;
    auto* indices = new unsigned int[num_inputs + 1];
    for (int i = 0; i < num_inputs + 1; i++) indices[i] = 0;
    while (num_finished < num_inputs) {
        unsigned min_i_input;
        Timestamp min_timestamp = LONG_LONG_MAX;
        for (int i = 1; i < num_inputs + 1; i++) {
            if (indices[i] + 1 >= capacities[i]) continue;

            const auto& transition = data[i][indices[i] + 1];
            if (transition.timestamp < min_timestamp) {
                min_timestamp = transition.timestamp;
                min_i_input = i;
            }
        }
        indices[min_i_input]++;
        data[0][indices[0]].timestamp = data[min_i_input][indices[min_i_input]].timestamp;
        data[0][indices[0]].value = logic_fn(data, num_inputs, indices, table, table_row_num);
        indices[0]++;
        if (indices[0] >= capacities[0]) break; // TODO handle overflow
        if (indices[min_i_input] >= capacities[min_i_input] - 1) num_finished++;
    }
    delete[] indices;
}
__host__ __device__ void single_input_algorithm(
        Transition** data, const unsigned int* capacities, char(*logic_fn)(char)
) {
    for (unsigned int i = 1; i < capacities[1]; i++) {
        if (i > capacities[0] - 1) break; // TODO handle overflow
        data[0][i - 1].timestamp = data[1][i].timestamp;
        data[0][i - 1].value = logic_fn(data[1][i].value);
    }
}
__host__ __device__ void and_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, and_logic);
}
__host__ __device__ void or_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, or_logic);
}
__host__ __device__ void xor_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, xor_logic);
}
__host__ __device__ void nand_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, nand_logic);
}
__host__ __device__ void nor_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, nor_logic);
}
__host__ __device__ void xnor_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, xnor_logic);
}
__host__ __device__ void not_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    single_input_algorithm(data, capacities, not_logic);
}
__host__ __device__ void buf_gate_fn(
        Transition** data,  // (capacity, num_inputs + num_outputs)
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    single_input_algorithm(data, capacities, buf_logic);
}

__host__ __device__ char primitive_logic(
        Transition** data, unsigned int num_inputs, const unsigned int* indices, char* table, unsigned int table_row_num
) {
    char output = 'x';  // if no matching rows, the output is x
    for (int i_table_row = 0; i_table_row < table_row_num; i_table_row++) {
        bool all_match = true;
        for (int i = 1; i < num_inputs + 1; i++) {
            auto value = data[i][indices[i]].value;
            value = value == 'z' ? 'x' : value;  // z is treated as x
            const auto& table_value = table[i_table_row * (num_inputs + 1) + (i - 1)];
            if (table_value != '?' and table_value != value) all_match = false;
        }
        if (all_match) {
            output = table[i_table_row * (num_inputs + 1) + num_inputs];
        }
    }
    return output;
}
__host__ __device__ void PrimitiveGate(
        Transition** data,
        const unsigned int* capacities,
        char* table, unsigned int table_row_num,
        unsigned int num_inputs, unsigned int num_outputs
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, primitive_logic);
};


__device__ GateFnPtr and_gate_fn_ptr = and_gate_fn;
__device__ GateFnPtr or_gate_fn_ptr = or_gate_fn;
__device__ GateFnPtr xor_gate_fn_ptr = xor_gate_fn;
__device__ GateFnPtr nand_gate_fn_ptr = nand_gate_fn;
__device__ GateFnPtr nor_gate_fn_ptr = nor_gate_fn;
__device__ GateFnPtr xnor_gate_fn_ptr = xnor_gate_fn;
__device__ GateFnPtr not_gate_fn_ptr = not_gate_fn;
__device__ GateFnPtr buf_gate_fn_ptr = buf_gate_fn;

