#include <cassert>

#include "builtin_gates.h"
#include "constants.h"


__host__ __device__ char and_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices , const char* table, const unsigned int table_row_num
) {
    bool is_all_one = true, has_zero = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        has_zero |= (value == '0');
        is_all_one &= (value == '1');
    }
    return has_zero ? '0' : (is_all_one ? '1' : 'x');
}

__host__ __device__ char or_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, const char* table, const unsigned int table_row_num
) {
    bool is_all_zero = true, has_one = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        has_one |= (value == '1');
        is_all_zero &= (value == '0');
    }
    return has_one ? '1' : (is_all_zero ? '0' : 'x');
}
__host__ __device__ char xor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, const char* table, const unsigned int table_row_num
) {
    char ret = '0';
    bool has_xz = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        has_xz |= (value == 'x' or value == 'z');
        ret = (ret == value) ? '0' : '1';
    }
    return has_xz ? 'x' : ret;
}
__host__ __device__ char nand_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, const char* table, unsigned int table_row_num
) {
    bool is_all_one = true, has_zero = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        has_zero |= (value == '0');
        is_all_one &= (value == '1');
    }
    return has_zero ? '1' : (is_all_one ? '0' : 'x');
}
__host__ __device__ char nor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, const char* table, const unsigned int table_row_num
) {
    bool is_all_zero = true, has_one = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        has_one |= (value == '1');
        is_all_zero &= (value == '0');
    }
    return has_one ? '0' : (is_all_zero ? '1' : 'x');
}
__host__ __device__ char xnor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, const char* table, const unsigned int table_row_num
) {
    char ret = '0';
    bool has_xz = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][indices[i]].value;
        has_xz |= (value == 'x' or value == 'z');
        ret = (ret == value) ? '0' : '1';
    }
    return has_xz ? 'x' : ret == '0' ? '1' : '0';
}
__host__ __device__ char not_logic(char v){ return (v == '0') ? '1' : ((v == '1') ? '0' : 'x'); }
__host__ __device__ char buf_logic(char v) { return  (v == 'z') ? 'x' : v; }

__host__ __device__ void merge_sort_algorithm(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs,
    LogicFn logic_fn,
    bool* overflow
) {
    if (data[1][0].value == 0) return;

    unsigned int indices[MAX_NUM_GATE_ARGS] = {0}; indices[0] = 1;
    assert(num_inputs + 1 <= MAX_NUM_GATE_ARGS);
    data[0][0].timestamp = data[1][0].timestamp;
    data[0][0].value = logic_fn(data, num_inputs, indices, table, table_row_num);

    unsigned int num_finished = 0;
    for (int i = 1; i < num_inputs + 1; i++) if (data[i][1].value == 0) num_finished++;

    // TODO shorter implementation
    while (num_finished < num_inputs) {
        // find min timestamp
        Timestamp min_timestamp = LONG_LONG_MAX;
        for (int i = 1; i < num_inputs + 1; i++) {
            if (indices[i] + 1 >= capacities[i] or data[i][indices[i] + 1].value == 0) continue;
            const auto& transition = data[i][indices[i] + 1];
            if (transition.timestamp < min_timestamp) min_timestamp = transition.timestamp;
        }

        // find advancing inputs, outputs
        bool advancing[MAX_NUM_GATE_ARGS] = { false };
        for (int i = 1; i < num_inputs + 1; i++) {
            if (indices[i] + 1 >= capacities[i] or data[i][indices[i] + 1].value == 0) continue;     // out of bound
            const auto& transition = data[i][indices[i] + 1];
            if (transition.timestamp == min_timestamp) {
                indices[i]++;
                advancing[i] = true;
                if (indices[i] + 1 >= capacities[i] or data[i][indices[i] + 1].value == 0) num_finished++;
            }
        }
        const char output_value = logic_fn(data, num_inputs, indices, table, table_row_num);
        for (int i = 1; i < num_inputs + 1; i++) {
            if (indices[0] >= capacities[0]) {
                *overflow = true;
                break;
            }
            if (not advancing[i]) continue;
            data[0][indices[0]].timestamp = min_timestamp; data[0][indices[0]].value = output_value;
            data[0][indices[0]].delay_info = data[i][indices[i]].delay_info; // propagate delay_info
            indices[0]++;
        }
        if (*overflow) break;
    }
}

__host__ __device__ void stepping_algorithm(Transition** data, const unsigned int* capacities, char(*logic_fn)(char)) {
    // single input doesn't overflow
    for (unsigned int i = 0; i < capacities[0]; i++) {
        data[0][i].value = data[1][i].value == 0 ? 0 : logic_fn(data[1][i].value);
        data[0][i].timestamp = data[1][i].timestamp;
        data[0][i].delay_info = data[1][i].delay_info;
    }
}
__host__ __device__ void and_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, and_logic, overflow_ptr);
}
__host__ __device__ void or_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacites,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    merge_sort_algorithm(data, capacites, table, table_row_num, num_inputs, or_logic, overflow_ptr);
}
__host__ __device__ void xor_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, xor_logic, overflow_ptr);
}
__host__ __device__ void nand_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs,
    bool* overflow_ptr
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, nand_logic, overflow_ptr);
}
__host__ __device__ void nor_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, nor_logic, overflow_ptr);
}
__host__ __device__ void xnor_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, xnor_logic, overflow_ptr);
}
__host__ __device__ void not_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    stepping_algorithm(data, capacities, not_logic);
}
__host__ __device__ void buf_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    stepping_algorithm(data, capacities, buf_logic);
}

__host__ __device__ char primitive_logic(
    Transition** data, unsigned int num_inputs, const unsigned int* indices, const char* table, unsigned int table_row_num
) {
//    TODO optimize
    char output = 'x';  // if no matching rows, the output is x
    for (int i_table_row = 0; i_table_row < table_row_num; i_table_row++) {
        bool all_match = true;
        for (int i = 1; i < num_inputs + 1; i++) {
            auto value = data[i][indices[i]].value;
            value = (value == 'z' ? 'x' : value);  // z is treated as x
            const auto& table_value = table[i_table_row * (num_inputs + 1) + (i - 1)];
            all_match &= (table_value == '?' or table_value == value);
        }
        output = all_match ? table[i_table_row * (num_inputs + 1) + num_inputs] : output;
    }
    return output;
}
__host__ __device__ void primitive_gate_fn(
    Transition** data,
    const unsigned int* capacities,
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
) {
    merge_sort_algorithm(data, capacities, table, table_row_num, num_inputs, primitive_logic, overflow_ptr);
};


__device__ GateFnPtr and_gate_fn_ptr = and_gate_fn;
__device__ GateFnPtr or_gate_fn_ptr = or_gate_fn;
__device__ GateFnPtr xor_gate_fn_ptr = xor_gate_fn;
__device__ GateFnPtr nand_gate_fn_ptr = nand_gate_fn;
__device__ GateFnPtr nor_gate_fn_ptr = nor_gate_fn;
__device__ GateFnPtr xnor_gate_fn_ptr = xnor_gate_fn;
__device__ GateFnPtr not_gate_fn_ptr = not_gate_fn;
__device__ GateFnPtr buf_gate_fn_ptr = buf_gate_fn;
__device__ GateFnPtr primitive_gate_fn_ptr = primitive_gate_fn;
