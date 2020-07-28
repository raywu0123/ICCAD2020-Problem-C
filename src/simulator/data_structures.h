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
    bool operator== (const DelayInfo& other) const {
        return arg == other.arg and edge_type == other.edge_type;
    }
};

enum class Values : char {
    PAD, ZERO, ONE, X, Z
};

inline std::ostream& operator<< (std::ostream& os, Values& v);

Values raw_to_enum(char r);
char enum_to_raw(Values v);

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
    unsigned int num_input, num_output;
    unsigned int table_row_num;
    Values* table;
};

struct ResourceBuffer {

    std::vector<bool*> overflows;
    std::vector<unsigned int> capacities;
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
    unsigned int* capacities;
    const ModuleSpec** module_specs;
    const SDFSpec** sdf_specs;
    Data* data_schedule;
    unsigned int num_modules;
};

#endif
