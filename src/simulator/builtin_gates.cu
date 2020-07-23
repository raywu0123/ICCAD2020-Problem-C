#include <cassert>

#include "builtin_gates.h"
#include "constants.h"


__host__ __device__ char and_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
) {
    bool is_all_one = true, has_zero = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][index].value;
        has_zero |= (value == '0');
        is_all_one &= (value == '1');
    }
    return has_zero ? '0' : (is_all_one ? '1' : 'x');
}

__host__ __device__ char or_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
) {
    bool is_all_zero = true, has_one = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][index].value;
        has_one |= (value == '1');
        is_all_zero &= (value == '0');
    }
    return has_one ? '1' : (is_all_zero ? '0' : 'x');
}
__host__ __device__ char xor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
) {
    char ret = '0';
    bool has_xz = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][index].value;
        has_xz |= (value == 'x' or value == 'z');
        ret = (ret == value) ? '0' : '1';
    }
    return has_xz ? 'x' : ret;
}
__host__ __device__ char nand_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, unsigned int table_row_num
) {
    bool is_all_one = true, has_zero = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][index].value;
        has_zero |= (value == '0');
        is_all_one &= (value == '1');
    }
    return has_zero ? '1' : (is_all_one ? '0' : 'x');
}
__host__ __device__ char nor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
) {
    bool is_all_zero = true, has_one = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][index].value;
        has_one |= (value == '1');
        is_all_zero &= (value == '0');
    }
    return has_one ? '0' : (is_all_zero ? '1' : 'x');
}
__host__ __device__ char xnor_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
) {
    char ret = '0';
    bool has_xz = false;
    for (int i = 1; i < num_inputs + 1; i++) {
        const auto& value = data[i][index].value;
        has_xz |= (value == 'x' or value == 'z');
        ret = (ret == value) ? '0' : '1';
    }
    return has_xz ? 'x' : ret == '0' ? '1' : '0';
}
__host__ __device__ char not_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
){
    const auto& v = data[1][index].value;
    return (v == '0') ? '1' : ((v == '1') ? '0' : 'x');
}
__host__ __device__ char buf_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
) {
    const auto& v = data[1][index].value;
    return  (v == 'z') ? 'x' : v;
}

__host__ __device__ void stepping_algorithm(
    Transition** data,
    LogicFn logic_fn,
    unsigned int num_inputs,
    const char* table, const unsigned int table_row_num
) {
    for (unsigned int i = 0; i < INITIAL_CAPACITY; i++) {
        data[0][i].value = data[1][i].value == 0 ? 0 : logic_fn(data, num_inputs, i, table, table_row_num);
        data[0][i].timestamp = data[1][i].timestamp;
        data[0][i].delay_info = data[1][i].delay_info;
    }
}
__host__ __device__ void and_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, and_logic, num_inputs, table, table_row_num);
}
__host__ __device__ void or_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, or_logic, num_inputs, table, table_row_num);
}
__host__ __device__ void xor_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, xor_logic, num_inputs, table, table_row_num);
}
__host__ __device__ void nand_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs
) {
    stepping_algorithm(data, nand_logic, num_inputs, table, table_row_num);
}
__host__ __device__ void nor_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, nor_logic, num_inputs, table, table_row_num);
}
__host__ __device__ void xnor_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, xnor_logic, num_inputs, table, table_row_num);
}
__host__ __device__ void not_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, not_logic, num_inputs, table, table_row_num);
}
__host__ __device__ void buf_gate_fn(
    Transition** data,  // (capacity, num_inputs + num_outputs)
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, buf_logic, num_inputs, table, table_row_num);
}

__host__ __device__ char primitive_logic(
    Transition** data, unsigned int num_inputs, const unsigned int index,
    const char* table, const unsigned int table_row_num
) {
//    TODO optimize
    char output = 'x';  // if no matching rows, the output is x
    for (int i_table_row = 0; i_table_row < table_row_num; i_table_row++) {
        bool all_match = true;
        for (int i = 1; i < num_inputs + 1; i++) {
            auto value = data[i][index].value;
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
    const char* table, const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
    stepping_algorithm(data, primitive_logic, num_inputs, table, table_row_num);

}


__device__ GateFnPtr and_gate_fn_ptr = and_gate_fn;
__device__ GateFnPtr or_gate_fn_ptr = or_gate_fn;
__device__ GateFnPtr xor_gate_fn_ptr = xor_gate_fn;
__device__ GateFnPtr nand_gate_fn_ptr = nand_gate_fn;
__device__ GateFnPtr nor_gate_fn_ptr = nor_gate_fn;
__device__ GateFnPtr xnor_gate_fn_ptr = xnor_gate_fn;
__device__ GateFnPtr not_gate_fn_ptr = not_gate_fn;
__device__ GateFnPtr buf_gate_fn_ptr = buf_gate_fn;
__device__ GateFnPtr primitive_gate_fn_ptr = primitive_gate_fn;
