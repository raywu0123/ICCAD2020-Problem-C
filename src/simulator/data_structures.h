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
    WaveSet();

    void summary() const;
    char* get_values_cuda();
    char* get_values_cpu() const;
    int size() const;

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
    __device__ virtual void compute(
            char** inputs, char** outputs,
            int num_inputs, int num_outputs, int stimuli_size
    ) const = 0;
};

class ANDGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class ORGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class NOTGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class NORGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class XORGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class XNORGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class NANDGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class BUFGate : public Gate {
    __device__ void compute(
        char** inputs, char** outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {};
};

class Primitive: public Gate {
public:
    Primitive(const vector<string> &vector_table, int input_size, int output_size);
    Primitive* cuda();

    __device__ void compute(
        char **inputs, char **outputs,
        int num_inputs, int num_outputs, int stimuli_size
    ) const override {
    };

    char* table{};    // on device
    int table_rows{};
};

class Module {
public:
    explicit Module(const vector<pair<Gate*, vector<int>>>& schedule);

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
    };

    Gate** submodule_schedule;
    int num_submodules;
    int* submodule_input_sizes;
    int* submodule_output_sizes;
};

#endif //ICCAD2020_DATA_STRUCTURES_H
