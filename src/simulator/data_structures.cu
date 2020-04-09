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

Module::Module(const vector<pair<Gate*, vector<int>>>& schedule) {
    num_submodules = schedule.size();
    submodule_schedule = new Gate*[num_submodules];
    submodule_input_sizes = new int[num_submodules];
    submodule_output_sizes = new int[num_submodules];
    for (int i = 0; i < num_submodules; i++) {
        submodule_schedule[i] = schedule[i].first;
        submodule_input_sizes[i] = schedule[i].second.size();
    }
}

Primitive::Primitive(const vector<string> &vector_table, int input_size, int output_size) {
    table_rows = vector_table.size();
    char host_table[(input_size + output_size) * table_rows];
    int i = 0;
    for(const auto& row : vector_table) {
        for(const auto& c : row) {
            host_table[i] = c;
            i++;
        }
    }
    cudaError_t result;
    result = cudaMalloc((void**) &table, sizeof(char) * (input_size + output_size) * table_rows);
    check_cuda_malloc(result);

    result = cudaMemcpy(table, host_table, sizeof(char) * (input_size + output_size) * table_rows, cudaMemcpyHostToDevice);
    check_cuda_memcpy(result);
}


Gate* Gate::cuda() const {
    cudaError_t result;
    Gate* device_ptr;
    result = cudaMalloc((void**) &device_ptr, sizeof(Gate));
    check_cuda_malloc(result);

    result = cudaMemcpy(device_ptr, this, sizeof(Gate), cudaMemcpyHostToDevice);
    check_cuda_memcpy(result);
    return device_ptr;
}
