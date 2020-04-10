#include <iostream>

#include "data_structures.h"

using namespace std;


void check_cuda_malloc(cudaError_t result) {
    if (result != cudaSuccess)
        cerr << "failed to allocate device memory" << endl;
}

void check_cuda_memcpy(cudaError_t result) {
    if (result != cudaSuccess)
        cerr << "Failed to copy to device memory" << endl;
}

char *WaveSet::get_values_cuda() {
    cudaError_t result;
    result = cudaMalloc((void**) &device_values, sizeof(char) * n_stimulus * stimuli_size);
    check_cuda_malloc(result);

    result = cudaMemcpy(device_values, values, sizeof(char) * n_stimulus * stimuli_size, cudaMemcpyHostToDevice);
    check_cuda_memcpy(result);
    return device_values;
}

char *WaveSet::get_values_cpu() const {
    cudaError_t result;
    result = cudaMemcpy(values, device_values, sizeof(char) * n_stimulus * stimuli_size, cudaMemcpyDeviceToHost);
    check_cuda_memcpy(result);
    return values;
}

int WaveSet::size() const {
    return n_stimulus * stimuli_size;
}

void WaveSet::summary() const {
    printf("n_stimulus: %d, stimuli_size: %d\n", n_stimulus, stimuli_size);
    printf("values:\n");
    for (int i = 0; i < n_stimulus; i++) {
        for (int j = 0; j < stimuli_size; j++) {
            cout << (int)values[stimuli_size * i + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

WaveSet::WaveSet() {
    values = (char*)malloc(sizeof(char) * n_stimulus * stimuli_size);
    timestamps = new size_t[n_stimulus * stimuli_size];
}
