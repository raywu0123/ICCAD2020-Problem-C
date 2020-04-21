#ifndef ICCAD2020_ACCUMULATORS_H
#define ICCAD2020_ACCUMULATORS_H

#include <vector>

#include "simulator/data_structures.h"

class Accumulator {
public:
    virtual ~Accumulator() = default;
    virtual void update(Transition*, unsigned int, unsigned int) = 0;
};

class VCDAccumulator : public Accumulator {
public:
    void update(Transition*, unsigned int, unsigned int) override;

    std::vector<Transition> transitions;
};

class SAIFAccumulator : public Accumulator {
public:
    void update(Transition *, unsigned int, unsigned int) ;
};

#endif
