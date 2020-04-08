#ifndef ICCAD2020_DATA_STRUCTURES_H
#define ICCAD2020_DATA_STRUCTURES_H


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <unordered_map>

#include "constants.h"


using namespace std;


class WaveSet {
public:
    WaveSet() {
        values = (char*)malloc(sizeof(char) * n_stimulus * stimuli_size);
        timestamps = new size_t[n_stimulus * stimuli_size];
    };

    void summary() const {
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

    char* get_values_cuda() {
        cudaError_t result;
        result = cudaMalloc((void**) &device_values, sizeof(char) * n_stimulus * stimuli_size);
        if (result != cudaSuccess)
            throw std::runtime_error("failed to allocate device memory");

        result = cudaMemcpy(device_values, values, sizeof(char) * n_stimulus * stimuli_size, cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
            throw std::runtime_error("failed to copy to device memory");
        return device_values;
    }

    char* get_values_cpu() const {
        cudaError_t result;
        result = cudaMemcpy(values, device_values, sizeof(char) * n_stimulus * stimuli_size, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
            throw std::runtime_error("failed to copy to device memory");
        return values;
    }

    int size() const {
        return n_stimulus * stimuli_size;
    }

    char* values = nullptr;
    char* device_values = nullptr;

    size_t* timestamps = nullptr;
    size_t* device_timestamps = nullptr;

    int n_stimulus{};
    int stimuli_size{};
};

class Gate {
public:
    // computes output for single stimuli
    __device__ void compute(
            char** inputs, char** outputs,
            int num_inputs, int num_outputs, int stimuli_size
    ) const;
    // TODO
};

class Primitive: public Gate {
public:
    Primitive(const vector<string>& vector_table, int input_size, int output_size)
            :input_size(input_size), output_size(output_size) {
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
        if (result != cudaSuccess)
            throw std::runtime_error("failed to allocate device memory");

        result = cudaMemcpy(table, host_table, sizeof(char) * (input_size + output_size) * table_rows, cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
            throw std::runtime_error("failed to copy to device memory");
    }

    Primitive* cuda() {
        cudaError_t result;
        Primitive* device_primitive;
        result = cudaMalloc((void**) &device_primitive, sizeof(Primitive));
        if (result != cudaSuccess)
            throw std::runtime_error("failed to allocate device memory");

        result = cudaMemcpy(device_primitive, this, sizeof(Primitive), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
            throw std::runtime_error("failed to copy to device memory");

        return device_primitive;
    }

    char* table{};    // on device
    int table_rows;
    int input_size, output_size;
};


class Module {
public:
    explicit Module(const vector<pair<Primitive*, vector<int>>>& schedule) {
        num_submodules = schedule.size();
        submodule_schedule = new Primitive*[num_submodules];
        submodule_input_sizes = new int[num_submodules];
        submodule_output_sizes = new int[num_submodules];
        for (int i = 0; i < num_submodules; i++) {
            submodule_schedule[i] = schedule[i].first;
            submodule_input_sizes[i] = schedule[i].second.size();
        }
    }

    // computes output for single stimuli
    __device__ void compute(char** data_schedule) const {
        int data_schedule_idx = 0;
        for(int i = 0; i < num_submodules; i++) {
            char** inputs = new char*[submodule_input_sizes[i]];
            for (int j = 0; j < submodule_input_sizes[i]; j++) {
                inputs[j] = data_schedule[data_schedule_idx];
                data_schedule_idx++;
            }

            char** outputs = new char*[submodule_output_sizes[i]];
            for (int j = 0; j < submodule_input_sizes[i]; j++) {
                inputs[j] = data_schedule[data_schedule_idx];
                data_schedule_idx++;
            }

            // submodule_schedule[i]->compute(inputs, outputs);
            // TODO
            delete[] inputs;
            delete[] outputs;
        }
    }

    Primitive** submodule_schedule;
    int num_submodules;
    int* submodule_input_sizes;
    int* submodule_output_sizes;
};

#endif //ICCAD2020_DATA_STRUCTURES_H
