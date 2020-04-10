#include "gtest/gtest.h"
#include <random>
#include "simulator/data_structures.h"
#include "simulator/builtin_gates.h"


void make_input(char** inputs, const int capacity, const int num_inputs) {
     for (int i = 0; i < num_inputs; i++) {
         for (int j = 0; j < capacity; j++) {
             int index = rand() % 4;
             switch (index) {
                 case 0:
                     inputs[i][j] = '0';
                     break;
                 case 1:
                     inputs[i][j] = '1';
                     break;
                 case 2:
                     inputs[i][j] = 'x';
                     break;
                 case 3:
                     inputs[i][j] = 'z';
                     break;
                 default:
                     break;
             }
         }
    }
}

void calculate_and_cpu(char** inputs, char* outputs, int capacity, int num_inputs) {

}

__global__ void test_kernel(
        GateFnPtr gate_fn_ptr,
        char** data,
        int num_inputs, int num_outputs,
        int capacity
) {
    int* capacities = new int[num_inputs + num_outputs];
    for(int i = 0; i < num_inputs + num_outputs; i++)
        capacities[i] = capacity;
//    (*gate_fn_ptr)(data, num_inputs, num_outputs, nullptr, capacities, 1);
}


TEST(gate_test, and_gate) {
    GateFnPtr host_gate_fn_ptr;
    cudaMemcpyFromSymbol(&host_gate_fn_ptr, and_gate_fn_ptr, sizeof(GateFnPtr));

    const int capacity = 100, num_inputs = 3;
    char** inputs = new char*[num_inputs];
    for (int i = 0; i < capacity; i++) {
        inputs[i] = new char[capacity];
    }

    char* expected_outputs = new char[capacity];

    make_input(inputs, capacity, num_inputs);

    char** device_data_ = new char*[num_inputs + 1];  // on host
    for (int i = 0; i < num_inputs + 1; i++) {
        cudaMalloc((void**)& device_data_[i], sizeof(char) * capacity);
        cudaMemcpy(device_data_[i], inputs[i], sizeof(char) * capacity, cudaMemcpyHostToDevice);
    }

    char** device_data;  // (num_inputs + num_outputs, capacity), on device
    cudaMalloc((void**)& device_data, sizeof(char*) * (num_inputs + 1));
    cudaMemcpy(device_data, device_data_, sizeof(char*) * (num_inputs + 1), cudaMemcpyHostToDevice);

    calculate_and_cpu(inputs, expected_outputs, capacity, num_inputs);
    test_kernel<<<1, 1>>>(host_gate_fn_ptr, device_data, num_inputs, 1, capacity);
    cudaDeviceSynchronize();

    char* host_outputs = new char[capacity];
    cudaMemcpy(host_outputs, device_data_[num_inputs], sizeof(char) * capacity, cudaMemcpyDeviceToHost);

    int error_num = 0;
    for (int i = 0; i < capacity; i++) {
        if (host_outputs[i] != expected_outputs[i])
            error_num++;
    }
    EXPECT_EQ(error_num, 0);

    delete[] device_data_;
}
