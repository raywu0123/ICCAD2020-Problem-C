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
    std::vector<std::string> args;
};

struct TokenInfo {
    std::string wire_name;
    BitWidth bitwidth;
    size_t bucket_index;
};

struct Transition {
    Timestamp timestamp;
    char value;
    Transition(Timestamp t, char v): timestamp(t), value(v) {};
};

struct Bucket {
    Wirekey wirekey;
    std::vector<Transition> transitions;
    Bucket(const std::string& wire_name, int bit_index): wirekey(Wirekey{wire_name, bit_index}) {};
    explicit Bucket(Wirekey  wirekey): wirekey(std::move(wirekey)) {};
};

typedef void (*GateFnPtr)(
        char** const data,  // (n_stimuli_parallel * capacity, num_inputs + num_outputs)
        int** const timestamps,
        const int num_inputs,
        const int num_outputs,
        const char* table,
        const int* capacities,
        const int n_stimuli_parallel
);

struct ModuleSpec{
    GateFnPtr* gate_schedule;
    int schedule_size;
    char** tables;
    int* num_inputs;
    int* num_outputs;
};

typedef void (*ModuleFnPtr)(
        const ModuleSpec& module_spec,
        Transition** const data_schedule,
        const int* data_schedule_offsets,  // offset to different gates' data
        const int* capacities,  // capacity of each gates' input/output
        const int* capacities_offsets,
        const int n_stimuli_parallel  // # stimuli computing in parallel, the whole system shares this parameter
);

struct BatchResource {

};

#endif
