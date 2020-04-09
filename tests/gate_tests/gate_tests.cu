#include "gtest/gtest.h"
#include <random>
#include "simulator/data_structures.h"


__global__ void and_gate_test_kernel(Gate* g, char** inputs, char* output, int stimuli_size, int num_inputs) {
    printf("hello %d %d\n", blockIdx.x, threadIdx.x);
    g->compute(inputs, nullptr, num_inputs, 1, stimuli_size);
    for (int i = 0; i < stimuli_size; i++) {
        output[i] = 'x';
    }
}

__global__ void hello_cuda() {
    printf("Hello CUDA!");
}

void make_input(char** inputs, int stimuli_size, int num_inputs) {
     for (int i_stimuli = 0; i_stimuli < stimuli_size; i_stimuli++) {
        for (int i_input = 0; i_input < num_inputs; i_input++) {
            int index = rand() % 4;
            switch (index) {
                case 0:
                    inputs[i_stimuli][i_input] = '0';
                    break;
                case 1:
                    inputs[i_stimuli][i_input] = '1';
                    break;
                case 2:
                    inputs[i_stimuli][i_input] = 'x';
                    break;
                case 3:
                    inputs[i_stimuli][i_input] = 'z';
            }
        }
    }
}

void calculate_and_cpu(char** inputs, char* outputs, int stimuli_size, int num_inputs) {
    for (int i_stimuli = 0; i_stimuli < stimuli_size; i_stimuli++) {
        bool is_all_one = true, has_zero = false;
        for (int i_input = 0; i_input < num_inputs; i_input++) {
            if (inputs[i_stimuli][i_input] == '0') {
                has_zero = true;
                break;
            } else if (inputs[i_stimuli][i_input] == 'x' or inputs[i_stimuli][i_input] == 'z') {
                is_all_one = false;
                break;
            }
        }
        if (has_zero) {
            outputs[i_stimuli] = '0';
        } else if (is_all_one) {
            outputs[i_stimuli] = '1';
        } else {
            outputs[i_stimuli] = 'x';
        }
    }
}


TEST(gate_test, and_gate) {
    printf("123\n\n");
    Gate* and_gate = ANDGate().cuda();

    const int stimuli_size = 100, num_inputs = 3;
    char** inputs = new char*[stimuli_size];
    char* expected_outputs = new char[stimuli_size];

    for (int i = 0; i < stimuli_size; i++) {
        inputs[i] = new char[num_inputs];
    }
    make_input(inputs, stimuli_size, num_inputs);

    char** device_inputs_ = new char*[stimuli_size];
    for (int i = 0; i < stimuli_size; i++) {
        cudaMalloc((void**)& device_inputs_[i], sizeof(char) * num_inputs);
        cudaMemcpy(device_inputs_[i], inputs[i], sizeof(char) * num_inputs, cudaMemcpyHostToDevice);
    }
    char** device_inputs;
    cudaMalloc((void**)& device_inputs, sizeof(char*) * stimuli_size);
    cudaMemcpy(device_inputs, device_inputs_, sizeof(char*) * stimuli_size, cudaMemcpyHostToDevice);
    delete[] device_inputs_;

    char* device_outputs;
    cudaMalloc((void**) &device_outputs, sizeof(char) * stimuli_size);

    calculate_and_cpu(inputs, expected_outputs, stimuli_size, num_inputs);
    and_gate_test_kernel<<<1, 1>>>(and_gate, device_inputs, device_outputs, stimuli_size, num_inputs);
    cudaDeviceSynchronize();

    char* host_outputs = new char[stimuli_size];
    cudaMemcpy(host_outputs, device_outputs, sizeof(char) * stimuli_size, cudaMemcpyDeviceToHost);

    int error_num = 0;
    for (int i = 0; i < stimuli_size; i++) {
        if (host_outputs[i] != 'x')
            error_num++;
    }
    EXPECT_EQ(error_num, 0);
}
