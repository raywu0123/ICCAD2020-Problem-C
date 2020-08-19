#include <cassert>

#include "memory_manager.h"
#include "utils.h"

using namespace std;

unordered_map<unsigned int, PointerHub*> MemoryManager::pointer_hubs;
mutex MemoryManager::mut;

void* MemoryManager::alloc(size_t size) {
    lock_guard<std::mutex> lock(mut);
    PointerHub* hub;
    auto it = pointer_hubs.find(size);
    if (it == pointer_hubs.end()) {
        hub = new PointerHub(size);
        pointer_hubs.emplace(size, hub);
    } else hub = it->second;

    return hub->get();
}

void MemoryManager::free(void* ptr, size_t size) {
    lock_guard<std::mutex> lock(mut);
    auto it = pointer_hubs.find(size);
    assert(it != pointer_hubs.end());

    auto* hub = it->second;
    hub->free(ptr);
}

void MemoryManager::finish() {
    for (auto& it : pointer_hubs) {
        it.second->finish();
        delete it.second;
    }
}

PointerHub::PointerHub(unsigned int size) : size(size) {}

void* PointerHub::get() {
    void* ptr;
    if (free_pointers.empty()) cudaMalloc(&ptr, size);
    else {
        ptr = *(free_pointers.begin());
        free_pointers.erase(ptr);
    }
    used_pointers.insert(ptr);
    return ptr;
}

void PointerHub::free(void* ptr) {
    assert(used_pointers.find(ptr) != used_pointers.end());
    used_pointers.erase(ptr);
    free_pointers.insert(ptr);
}

void PointerHub::finish() {
    for (auto& ptr : free_pointers) cudaFree(ptr);
    for (auto& ptr : used_pointers) cudaFree(ptr);
}
