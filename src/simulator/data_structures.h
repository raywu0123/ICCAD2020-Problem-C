#ifndef ICCAD2020_DATA_STRUCTURES_H
#define ICCAD2020_DATA_STRUCTURES_H

#include <vector>
#include <string>
#include <ostream>

#include "constants.h"


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
    std::vector<NUM_ARG_TYPE> args;
};

struct SDFSpec {
    unsigned int num_rows;
    NUM_ARG_TYPE *input_index, *output_index;
    char* edge_type;
    int *rising_delay, *falling_delay;
};

struct TokenInfo {
    std::string wire_name;
    BitWidth bitwidth;
    size_t bucket_index;
};


enum class Values : char {
    PAD, ZERO, ONE, X, Z
};

inline std::ostream& operator<< (std::ostream& os, Values& v);

Values raw_to_enum(char r);
char enum_to_raw(Values v);


enum class EdgeTypes : char {
    UNDEF, RISING, FALLING, XZ, ZX, NODELAY
};
__host__ __device__ char edge_type_to_raw(EdgeTypes);
__host__ __device__ EdgeTypes raw_to_edge_type(char);
__host__ __device__ EdgeTypes get_edge_type(const Values& v1, const Values& v2);


struct DelayInfo {
    DelayInfo() = default;
    DelayInfo(NUM_ARG_TYPE arg, char edge_type) : arg(arg), edge_type(raw_to_edge_type(edge_type)) {};
    NUM_ARG_TYPE arg = 0;
    EdgeTypes edge_type = EdgeTypes::UNDEF;
    bool operator== (const DelayInfo& other) const {
        return arg == other.arg and edge_type == other.edge_type;
    }
};


struct Transition {
    Timestamp timestamp = 0;
    Values value = Values::PAD;
    Transition() = default;
    Transition(Timestamp t, Values v): timestamp(t), value(v) {};
    Transition(Timestamp t, char r): timestamp(t), value(raw_to_enum(r)) {};

    bool operator== (const Transition& other) const {
        return timestamp == other.timestamp and value == other.value;
    }
    bool operator!= (const Transition& other) const {
        return not operator==(other);
    }
};

struct Data {
    Transition* transitions = nullptr;
    unsigned int* size = nullptr;
};

std::ostream& operator<< (std::ostream& os, const Transition& transition);

struct ModuleSpec{
    NUM_ARG_TYPE num_input, num_output;
    Values* table;
};

struct ResourceBuffer {

    std::vector<bool*> overflows;
    std::vector<CAPACITY_TYPE> capacities;
    std::vector<const ModuleSpec*> module_specs;
    std::vector<const SDFSpec*> sdf_specs;
    std::vector<Data> data_schedule;

    ResourceBuffer ();
    void finish_module();
    unsigned int size = 0;
};


struct BatchResource {
    void init(const ResourceBuffer&);
    void free() const;

    bool** overflows;
    CAPACITY_TYPE* capacities;
    const ModuleSpec** module_specs;
    const SDFSpec** sdf_specs;
    Data* data_schedule;
    unsigned int num_modules;
};

#endif
