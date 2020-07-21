#include <iostream>

#include "memory_manager.h"
#include "utils.h"

using namespace std;

Transition* MemoryManager::alloc(size_t size) {
//    TODO buddy tree
    Transition* p;
    cudaErrorCheck(cudaMalloc((void**) &p, sizeof(Transition) * size));
    return p;
}

void MemoryManager::free(Transition* p) {
    cudaErrorCheck(cudaFree(p));
}
