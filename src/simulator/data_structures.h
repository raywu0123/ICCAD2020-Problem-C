#ifndef ICCAD2020_DATA_STRUCTURES_H
#define ICCAD2020_DATA_STRUCTURES_H

#include <vector>
#include <string>
#include <ostream>


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


struct DelayInfo {
    DelayInfo() = default;
    DelayInfo(unsigned int arg, char edge_type) : arg(arg), edge_type(edge_type) {};
    unsigned int arg = 0;
    char edge_type = 0;
};

struct Transition {
    Timestamp timestamp ;
    char value;
    DelayInfo delay_info;
    explicit Transition(): timestamp(0), value(0) {};
    Transition(Timestamp t, char v): timestamp(t), value(v) {};
    Transition(Timestamp t, char v, DelayInfo d): timestamp(t), value(v), delay_info(d) {};

    bool operator== (const Transition& other) const {
        return timestamp == other.timestamp and value == other.value;
    }
    bool operator!= (const Transition& other) const {
        return not operator==(other);
    }
};

std::ostream& operator<< (std::ostream& os, const Transition& transition);

typedef void (*GateFnPtr)(
    Transition** data,  // (n_stimuli_parallel * capacity, num_inputs + num_outputs)
    const unsigned int* capacities,
    const char* table,
    const unsigned int table_row_num,
    const unsigned int num_inputs, const unsigned int num_outputs,
    bool* overflow_ptr
);

typedef char (*LogicFn)(Transition**, unsigned int, const unsigned int*, const char* table, const unsigned int table_row_num);
struct ModuleSpec{
    GateFnPtr* gate_schedule;
    unsigned int schedule_size; // number of gates
    unsigned int num_module_input, num_module_output, num_module_args;
    char** tables;
    unsigned int* table_row_num;
    unsigned int* gate_specs;
    unsigned int* num_inputs;  // how many inputs for every gate
    unsigned int* num_outputs;  // how many outputs for every gate, currently assume its always 1
};

struct ResourceBuffer {
    std::vector<const ModuleSpec*> module_specs;
    std::vector<const SDFSpec*> sdf_specs;
    std::vector<Transition*> data_schedule;
    std::vector<unsigned int> data_schedule_offsets;
    std::vector<unsigned int> capacities;
    std::vector<bool*> overflows;

    ResourceBuffer ();
    int size() const;
};


struct BatchResource {
    void init(const ResourceBuffer&);
    void free() const;

    const ModuleSpec** module_specs;
    const SDFSpec** sdf_specs;
    Transition** data_schedule;
    unsigned int* capacities;
    unsigned int* data_schedule_offsets; // offsets to each module
    unsigned int num_modules;
    bool** overflows;
};

#endif
