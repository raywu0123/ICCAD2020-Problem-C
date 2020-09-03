#include <iostream>
#include <cassert>

#include "memory_manager.h"

using namespace std;

unordered_map<unsigned int, PointerHub*> MemoryManager::pointer_hubs;
unordered_map<unsigned int, PointerHubHost*> MemoryManager::pointer_hubs_host;

void* MemoryManager::alloc(size_t size) {
    PointerHub* hub;
    auto it = pointer_hubs.find(size);
    if (it == pointer_hubs.end()) {
        hub = new PointerHub(size);
        pointer_hubs.emplace(size, hub);
    } else hub = it->second;

    return hub->get();
}

void MemoryManager::free(void* ptr, size_t size) {
    auto it = pointer_hubs.find(size);
    assert(it != pointer_hubs.end());

    auto* hub = it->second;
    hub->free(ptr);
}

void* MemoryManager::alloc_host(size_t size) {
    PointerHubHost* hub;
    auto it = pointer_hubs_host.find(size);
    if (it == pointer_hubs_host.end()) {
        hub = new PointerHubHost(size);
        pointer_hubs_host.emplace(size, hub);
    } else hub = it->second;

    return hub->get();
}

void MemoryManager::free_host(void* ptr, size_t size) {
    auto it = pointer_hubs_host.find(size);
    assert(it != pointer_hubs_host.end());

    auto* hub = it->second;
    hub->free(ptr);
}

void MemoryManager::finish() {
    for(auto& it : pointer_hubs) {
        it.second->finish();
        delete it.second;
    }
    for(auto& it : pointer_hubs_host) {
        it.second->finish();
        delete it.second;
    }
}

PointerHub::PointerHub(unsigned int size) :size(size) {}

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

PointerHubHost::PointerHubHost(unsigned int size) : size(size) {}

void* PointerHubHost::get() {
    void* ptr;
    if (free_pointers.empty()) cudaMallocHost(&ptr, size);
    else {
        ptr = *(free_pointers.begin());
        free_pointers.erase(ptr);
    }
    used_pointers.insert(ptr);
    return ptr;
}

void PointerHubHost::free(void* ptr) {
    assert(used_pointers.find(ptr) != used_pointers.end());
    used_pointers.erase(ptr);
    free_pointers.insert(ptr);
}

void PointerHubHost::finish() {
    for (auto& ptr : free_pointers) cudaFreeHost(ptr);
    for (auto& ptr : used_pointers) cudaFreeHost(ptr);
}
