#include "wire.h"

using namespace std;

Wire::Wire(const WireInfo& wire_info) {
    wire_infos.push_back(wire_info);
}

// memory pattern
//      0                   1   2   3  ...    C
// 0   previous_transition t00 t01 t02 ... t0(c-1)
// 1   t0(c-1)             t10 t11 t12
// 2   t1(c-1)
// .
// .
// N-1


void Wire::assign(const Wire& other_wire) {
    wire_infos.insert(wire_infos.end(), other_wire.wire_infos.begin(), other_wire.wire_infos.end());
}

void Wire::load_from_bucket(
    Transition* ptr, unsigned int capacity,
    unsigned int stimuli_index, unsigned int bucket_index, unsigned int size
) {
    cudaMemcpy(
        ptr + capacity * stimuli_index,
        bucket.transitions.data() + bucket_index,
        sizeof(Transition) * size,
        cudaMemcpyHostToDevice
    );
}

void Wire::store_to_bucket(const vector<Transition*>& data_ptrs, unsigned int capacity) {
    for (const auto& data_ptr : data_ptrs) bucket.push_back(data_ptr, capacity);
}

ConstantWire::ConstantWire(char value): value(value) {
    bucket.transitions.clear();
    bucket.transitions.emplace_back(0, value);
}
