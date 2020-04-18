#include "wire.h"

using namespace std;

void Wire::set_input(
        const std::vector<Transition>& ts,
        const std::vector<unsigned int>& stimuli_edges,
        unsigned int i_stimuli
) {
    alloc();
    unsigned int start_index = stimuli_edges[i_stimuli];
    unsigned int end_index = stimuli_edges[i_stimuli + 1];
    cudaMemcpy(
        data_ptr + sizeof(Transition) * i_stimuli,
        &ts[0 + start_index],
        (end_index - start_index) * sizeof(Transition),
        cudaMemcpyHostToDevice
    );
}

void Wire::alloc() {
//        called by Cell::prepare_resource or Wire::set_input
    if (data_ptr != nullptr) // prevent double-alloc
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
