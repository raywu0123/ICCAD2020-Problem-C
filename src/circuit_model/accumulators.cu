#include "accumulators.h"


void VCDAccumulator::update(Transition* data, unsigned int capacity, unsigned int n_stimuli_parallel) {
    for (unsigned i_stimuli = 0; i_stimuli < n_stimuli_parallel; i_stimuli++) {
        for (unsigned int i = 1; i < capacity; i++) {  // starting from 1 because 0 is for initial value
            const auto& transition = data[i_stimuli * capacity + i];
            if (transition.value == 0) break;
            transitions.push_back(transition);
        }
    }
}

void SAIFAccumulator::update(Transition *, unsigned int, unsigned int) {

}
