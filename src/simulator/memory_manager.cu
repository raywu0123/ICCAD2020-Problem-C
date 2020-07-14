#include <iostream>

#include "memory_manager.h"

using namespace std;

Transition* MemoryManager::alloc(size_t size) {
//    TODO buddy tree
    Transition* p;
    auto status = cudaMalloc((void**) &p, sizeof(Transition) * size);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return p;
}

void MemoryManager::free(Transition* p) {
    auto status = cudaFree(p);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
