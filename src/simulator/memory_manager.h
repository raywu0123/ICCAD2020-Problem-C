#ifndef ICCAD2020_MEMORY_MANAGER_H
#define ICCAD2020_MEMORY_MANAGER_H

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
    PinnedMemoryAllocator(const PinnedMemoryAllocator&) {}
    ~PinnedMemoryAllocator() {};
    template<class U>
    PinnedMemoryAllocator(const PinnedMemoryAllocator<U>& other) {}

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

class MemoryManager {
public:
    static Transition* alloc(size_t size);
    static void free(Transition*);

private:
    static Transition* memory;
};


#endif
