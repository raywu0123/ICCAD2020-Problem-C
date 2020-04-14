#include "wire.h"

using namespace std;

void Wire::set_input(const vector<Transition> &ts, unsigned int start_index, unsigned int size) {
    alloc();
    cudaMemcpy(data_ptr, &ts[0 + start_index], size * sizeof(Transition), cudaMemcpyHostToDevice);
}

void Wire::alloc() {
//        called by Cell::prepare_resource or Wire::set_input
    if (data_ptr == nullptr)
        return;
    data_ptr = MemoryManager::alloc(capacity * N_STIMULI_PARALLEL);
}

void Wire::free()  {
    accumulator->update();
    MemoryManager::free(data_ptr);
    data_ptr = nullptr;
};


ConstantWire::ConstantWire(char value): value(value) {
    alloc();
    Transition t{0, value};
    cudaMemcpy(data_ptr, &t, sizeof(Transition), cudaMemcpyHostToDevice);
};
