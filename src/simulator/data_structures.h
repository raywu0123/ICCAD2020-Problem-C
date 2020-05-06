#ifndef ICCAD2020_DATA_STRUCTURES_H
#define ICCAD2020_DATA_STRUCTURES_H

#include <vector>
#include <string>


typedef std::pair<int, int> BitWidth;
typedef std::pair<std::string, int> Wirekey;
typedef long long int Timestamp;

struct pair_hash {
    template<class T1, class T2>
    std:: size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

struct SubmoduleSpec {
    std::string name;
    std::string type;
    std::vector<unsigned int> args;
};

struct SDFSpec {
    unsigned int num_rows;
    unsigned int *input_index, *output_index;
    char* edge_type;
    int *rising_delay, *falling_delay;
};

struct TokenInfo {
    std::string wire_name;
    BitWidth bitwidth;
    size_t bucket_index;
};

struct Transition {
    Timestamp timestamp ;
    char value;
    explicit Transition(): timestamp(0), value(0) {};
    Transition(Timestamp t, char v): timestamp(t), value(v) {};
};

typedef void (*GateFnPtr)(
    Transition** data,  // (n_stimuli_parallel * capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    char* table,
    unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs
);

typedef char (*LogicFn)(Transition**, unsigned int, const unsigned int*, char* table, unsigned int table_row_num);
struct ModuleSpec{
    GateFnPtr* gate_schedule;
    unsigned int schedule_size; // number of gates
    unsigned int data_schedule_size = 0;  // number of wires in the whole schedule
    unsigned int* data_schedule_indices;
    unsigned int num_module_input, num_module_output;
    char** tables;
    unsigned int* table_row_num;
    unsigned int* num_inputs;  // how many inputs for every gate
    unsigned int* num_outputs;  // how many outputs for every gate, currently assume its always 1
};

struct BatchResource {
    const ModuleSpec** module_specs;
    const SDFSpec** sdf_specs;
    Transition** data_schedule;
    unsigned int* capacities;
    unsigned int* data_schedule_offsets; // offsets to each module
    unsigned int num_modules;
};

struct ResourceBuffer {
    std::vector<const ModuleSpec*> module_specs;
    std::vector<const SDFSpec*> sdf_specs;
    std::vector<const Transition*> data_schedule;
    std::vector<unsigned int> data_schedule_offsets;
    std::vector<unsigned int> capacities;

    void clear() {
        module_specs.clear();
        sdf_specs.clear();
        data_schedule.clear();
        data_schedule_offsets.clear();
        capacities.clear();
    }

    int size() const { return module_specs.size(); }

    BatchResource get_batch_resource() {
        BatchResource batch_resource{};
        unsigned int num_modules = size();
        batch_resource.num_modules = num_modules;

        cudaMalloc((void**) &batch_resource.module_specs, sizeof(ModuleSpec*) * num_modules);
        cudaMalloc((void**) &batch_resource.sdf_specs, sizeof(SDFSpec*) * num_modules);
        cudaMalloc((void**) &batch_resource.data_schedule, sizeof(Transition*) * data_schedule.size());
        cudaMalloc((void**) &batch_resource.data_schedule_offsets, sizeof(unsigned int) * num_modules);
        cudaMalloc((void**) &batch_resource.capacities, sizeof(unsigned int) * capacities.size());

        cudaMemcpy(batch_resource.module_specs, module_specs.data(), sizeof(ModuleSpec*) * num_modules, cudaMemcpyHostToDevice);
        cudaMemcpy(batch_resource.sdf_specs, sdf_specs.data(), sizeof(SDFSpec*) * num_modules, cudaMemcpyHostToDevice);
        cudaMemcpy(batch_resource.data_schedule, data_schedule.data(), sizeof(Transition*) * data_schedule.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(batch_resource.data_schedule_offsets, data_schedule_offsets.data(), sizeof(unsigned int) * num_modules, cudaMemcpyHostToDevice);
        cudaMemcpy(batch_resource.capacities, capacities.data(), sizeof(unsigned int) * capacities.size(), cudaMemcpyHostToDevice);
        clear();

        return batch_resource;
    }
};

#endif
