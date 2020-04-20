#include <iostream>
#include "wire.h"

using namespace std;

// memory pattern
//      0                   1   2   3  ...    C
// 0   previous_transition t00 t01 t02 ... t0(c-1)
// 1   t0(c-1)             t10 t11 t12
// 2   t1(c-1)
// .
// .
// N-1

// previous_transition <- t(N-1)(c-1)
void Wire::set_input(
    const std::vector<Transition>& ts,
    const std::vector<unsigned int>& stimuli_edges,
    unsigned int i_stimuli
) {
    alloc();
    unsigned int start_index = stimuli_edges[i_stimuli];
    unsigned int end_index = stimuli_edges[i_stimuli + 1];
    if (end_index - start_index >= capacity - 1) throw runtime_error("Stimuli size too large");

    cudaMemcpy(
        &data_ptr[capacity * i_stimuli + 1],
        &ts[start_index],
        (end_index - start_index) * sizeof(Transition),
        cudaMemcpyHostToDevice
    );
    if (i_stimuli == N_STIMULI_PARALLEL - 1) {
        previous_transition = ts[end_index - 1];
    }
    else {
        cudaMemcpy(
            &data_ptr[capacity * (i_stimuli + 1)],
            &ts[end_index - 1],
            sizeof(Transition),
            cudaMemcpyHostToDevice
        );
    }
    cudaDeviceSynchronize();
}

void Wire::alloc() {
//        called by Cell::prepare_resource or Wire::set_input
    if (data_ptr != nullptr) // prevent double-alloc
        return;
    data_ptr = MemoryManager::alloc(capacity * N_STIMULI_PARALLEL);
    cudaMemcpy(&data_ptr[0], &previous_transition, sizeof(Transition), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void Wire::free()  {
    auto host_data_ptr = new Transition[capacity * N_STIMULI_PARALLEL];
    cudaMemcpy(host_data_ptr, data_ptr, sizeof(Transition) * capacity * N_STIMULI_PARALLEL, cudaMemcpyDeviceToHost);
    accumulator->update(host_data_ptr, capacity, N_STIMULI_PARALLEL);
    delete[] host_data_ptr;

    MemoryManager::free(data_ptr);
    data_ptr = nullptr;
}


ConstantWire::ConstantWire(char value): value(value) {
    alloc();
    Transition t{0, value};
    cudaMemcpy(data_ptr, &t, sizeof(Transition), cudaMemcpyHostToDevice);
}
