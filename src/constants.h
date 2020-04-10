#ifndef ICCAD2020_CONSTANTS_H
#define ICCAD2020_CONSTANTS_H

#include <vector>

using namespace std;


typedef pair<int, int> BitWidth;
typedef pair<string, int> Wirekey;

struct pair_hash {
    template<class T1, class T2>
    std:: size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};


typedef long long int Timestamp;

const int NUM_VALUES = 4;
const char VALUES[NUM_VALUES] = {'0', '1', 'x', 'z'};

enum STD_CELL_DECLARE_TYPE {
    STD_CELL_INPUT,
    STD_CELL_OUTPUT,
    STD_CELL_WIRE,
    STD_CELL_SUPPLY1,
    STD_CELL_SUPPLY0,
    STD_CELL_LAST=STD_CELL_SUPPLY0,
};

const STD_CELL_DECLARE_TYPE STD_CELL_DECLARE_TYPES[] = {
    STD_CELL_INPUT, STD_CELL_OUTPUT, STD_CELL_WIRE, STD_CELL_SUPPLY1, STD_CELL_SUPPLY0
};

struct StdCellDeclare {
    vector<vector<string>> buckets{5};
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
    char** const data_schedule,
    int** const timestamp_schedule,
    const int* data_schedule_offsets,  // offset to different gates' data
    const int* capacities,  // capacity of each gates' input/output
    const int* capacities_offsets,
    const int n_stimuli_parallel  // # stimuli computing in parallel, the whole system shares this parameter
);
#endif //ICCAD2020_CONSTANTS_H
