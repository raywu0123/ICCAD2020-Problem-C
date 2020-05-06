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

Transition* Wire::alloc() {
    auto* data_ptr = MemoryManager::alloc(capacity);
    data_ptrs.emplace_back(data_ptr, capacity);
    return data_ptr;
}

void Wire::free() {
    for (const auto& data_ptr : data_ptrs) {
        MemoryManager::free(data_ptr.ptr);
    }
    data_ptrs.clear();
}

void Wire::load_from_bucket(unsigned int index, unsigned int size) {
    cudaMemcpy(
        data_ptrs.back().ptr,
        bucket.transitions.data() + index,
        sizeof(Transition) * size,
        cudaMemcpyHostToDevice
    );
}

void Wire::store_to_bucket() {
    for (const auto& data_ptr : data_ptrs) bucket.push_back(data_ptr);
}

ConstantWire::ConstantWire(char value): value(value) {
    bucket.transitions.emplace_back(0, value);
}
