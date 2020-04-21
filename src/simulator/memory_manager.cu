#include "memory_manager.h"

void MemoryManager::init() {

}

Transition* MemoryManager::alloc(size_t size) {
//    TODO buddy tree
    Transition* p;
    cudaMalloc((void**) &p, sizeof(Transition) * size);
    cudaMemset(&p, 0, sizeof(Transition) * size);
    return p;
}

void MemoryManager::free(Transition* p) {
    cudaFree(p);
}
