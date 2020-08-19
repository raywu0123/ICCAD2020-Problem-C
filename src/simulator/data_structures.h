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
    DelayInfo(unsigned int arg, char edge_type) : arg(arg), edge_type(raw_to_edge_type(edge_type)) {};
    unsigned int arg = 0;
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

    Data() = default;
    Data(void* p1, void* p2) : transitions((Transition*) p1), size((unsigned int*) p2) {}
};

std::ostream& operator<< (std::ostream& os, const Transition& transition);

struct ModuleSpec{
    unsigned int num_input, num_output;
    unsigned int table_row_num;
    Values* table;
};

#endif
