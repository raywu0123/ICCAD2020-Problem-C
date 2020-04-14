#ifndef ICCAD2020_MEMORY_MANAGER_H
#define ICCAD2020_MEMORY_MANAGER_H

#include <constants.h>
#include <simulator/data_structures.h>


class MemoryManager {
public:
    static void init();
    static Transition* alloc(size_t size);
    static void free(Transition*);

private:
    static Transition* memory;
};


#endif
