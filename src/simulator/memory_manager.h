#ifndef ICCAD2020_MEMORY_MANAGER_H
#define ICCAD2020_MEMORY_MANAGER_H

#include <unordered_set>
#include <unordered_map>

#include <constants.h>
#include <simulator/data_structures.h>

template <typename T>
class PinnedMemoryAllocator {
public:
    typedef T                   value_type;
    typedef value_type*         pointer;
    typedef value_type&         reference;
    typedef value_type const*   const_pointer;
    typedef value_type const&   const_reference;
    typedef size_t              size_type;
    typedef ptrdiff_t           difference_type;

    PinnedMemoryAllocator() = default;
    PinnedMemoryAllocator(const PinnedMemoryAllocator&) = default;
    ~PinnedMemoryAllocator() = default;;
    template<class U>
    explicit PinnedMemoryAllocator(const PinnedMemoryAllocator<U>& other) {}

    template<class U>
    struct rebind { using other = PinnedMemoryAllocator<U>; };

    template<class U>
    bool operator== (PinnedMemoryAllocator<U> const&) const { return true; }

    template<class U>
    bool operator!= (PinnedMemoryAllocator<U> const&) const { return false; }

    pointer allocate(size_type n) {
        T* t;
        cudaMallocHost((void**) &t, sizeof(T) * n);
        return t;
    }
    void deallocate(void* p, size_type) {
        if (p) cudaFreeHost(p);
    }

    pointer address(reference x) { return &x; }
    const_pointer address(const_reference x) { return &x; }
    size_type max_size() const { return size_t(-1); }
};

template<typename T> using PinnedMemoryVector = std::vector<T, PinnedMemoryAllocator<T>>;


class PointerHub {
public:
    explicit PointerHub(unsigned int size);
    std::unordered_set<void*> free_pointers, used_pointers;
    unsigned int size = 0;

    void* get();
    void free(void*);
    void finish();
};


class PointerHubHost {
public:
    explicit PointerHubHost(unsigned int size);
    std::unordered_set<void*> free_pointers, used_pointers;
    unsigned int size = 0;

    void* get();
    void free(void*);
    void finish();
};


class MemoryManager {
public:
    static std::unordered_map<unsigned int, PointerHub*> pointer_hubs;
    static std::unordered_map<unsigned int, PointerHubHost*> pointer_hubs_host;

    static void* alloc(size_t size);
    static void free(void*, size_t);
    static void* alloc_host(size_t size);
    static void free_host(void*, size_t);
    static void finish();

private:
    static Transition* memory;
};


#endif
