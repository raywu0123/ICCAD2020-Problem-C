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
    // Multiple alloc() calls would happen when same wire is used as input for different cells
    auto* data_ptr = MemoryManager::alloc(capacity * N_STIMULI_PARALLEL);
    data_ptrs.emplace(data_ptr, capacity);
    return data_ptr;
}

void Wire::free(Transition* data_ptr) {
    // Using data_ptrs vector to records
    // repeated calling free() would not cause a problem
    if (data_ptrs.find(data_ptr) != data_ptrs.end()) {
        MemoryManager::free(data_ptr);
        data_ptrs.erase(data_ptr);
    }
}

void Wire::load_from_bucket(Transition* ptr, unsigned int stimuli_index, unsigned int bucket_index, unsigned int size) {
    cudaMemcpy(
        ptr + capacity * stimuli_index,
        bucket.transitions.data() + bucket_index,
        sizeof(Transition) * size,
        cudaMemcpyHostToDevice
    );
}

void Wire::store_to_bucket() {
    for (const auto& item : data_ptrs) bucket.push_back(item.first, item.second);
}

void Wire::reset_capacity() {
    capacity = INITIAL_CAPACITY;
}

Transition* Wire::increase_capacity() {
    if (data_ptrs.size() != 1) throw runtime_error("Encountered multiple data_ptrs while increasing capacity.");
    free(data_ptrs.begin()->first);
    capacity *= 2;
    return alloc();
}

ConstantWire::ConstantWire(char value): value(value) {
    bucket.transitions.emplace_back(0, value);
}
