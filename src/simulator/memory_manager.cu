#include <iostream>

#include "memory_manager.h"
#include "utils.h"

using namespace std;

Data MemoryManager::alloc(size_t size) {
//    TODO buddy tree
    Transition* t;
    cudaErrorCheck(cudaMalloc((void**) &t, sizeof(Transition) * size));
    unsigned int* i;
    cudaErrorCheck(cudaMalloc((void**) &i, sizeof(unsigned int)));
    return {t, i};
}

void MemoryManager::free(Data d) {
    cudaErrorCheck(cudaFree(d.transitions));
    cudaErrorCheck(cudaFree(d.size));
}
