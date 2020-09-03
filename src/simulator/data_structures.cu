#include <iostream>
#include <cassert>

#include "constants.h"
#include "data_structures.h"

using namespace std;

__host__ __device__ EdgeTypes get_edge_type(const Values& v1, const Values& v2) {
    if (v2 == Values::ONE or v1 == Values::ZERO) return EdgeTypes::RISING;
    if (v2 == Values::ZERO or v1 == Values::ONE) return EdgeTypes::FALLING;

    if (v1 == Values::X and v2 == Values::Z) return EdgeTypes::XZ;
    if (v1 == Values::Z and v2 == Values::X) return EdgeTypes::ZX;
    return EdgeTypes::UNDEF;
}

__host__ __device__ char edge_type_to_raw(EdgeTypes e) {
    switch (e) {
        case EdgeTypes::RISING:
            return '+';
        case EdgeTypes::FALLING:
            return '-';
        default:
            return 'x';
    }
}
__host__ __device__ EdgeTypes raw_to_edge_type(char r) {
    switch (r) {
        case '+':
            return EdgeTypes::RISING;
        case '-':
            return EdgeTypes::FALLING;
        default:
            return EdgeTypes::UNDEF;
    }
}

Values raw_to_enum(char v) {
    switch (v) {
        case '0':
            return Values::ZERO;
        case '1':
            return Values::ONE;
        case 'x':
        case 'X':
            return Values::X;
        case 'z':
        case 'Z':
            return Values::Z;
        default:
            return Values::PAD;
    }
}

char enum_to_raw(Values v) {
    switch (v) {
        case Values::ZERO:
            return '0';
        case Values::ONE:
            return '1';
        case Values::X:
            return 'x';
        case Values::Z:
            return 'z';
        default:
            return '_';
    }
}

inline std::ostream& operator<< (std::ostream& os, const Values& v) {
    os << enum_to_raw(v);
    return os;
}

std::ostream& operator<< (std::ostream& os, const Transition& transition) {
    os << "(" << transition.timestamp << ", " << transition.value << ")";
    return os;
}
