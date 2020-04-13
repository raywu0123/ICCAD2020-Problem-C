#ifndef ICCAD2020_MEMORY_MANAGER_H
#define ICCAD2020_MEMORY_MANAGER_H

#include <constants.h>

struct WireData {
    char* values;
    Timestamp* timestamps;
};


class MemoryManager {
public:
    static void init();
    static WireData* alloc(size_t size);
    static void free(WireData*);

private:
    static WireData* memory;
};


#endif
