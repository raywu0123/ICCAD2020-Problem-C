#include "simulation_result.h"


VCDResult::VCDResult(Circuit &circuit) : SimulationResult(circuit) {
    for (const auto& it : circuit.wires) {
        const auto& accumulator = new VCDAccumulator();
        it->accumulator = accumulator;
        accumulators.push_back(accumulator);
    }
}

void VCDResult::write(char *path) {
// TODO
// Merge sort accumulators
}

SAIFResult::SAIFResult(Circuit &circuit) : SimulationResult(circuit) {
    for (const auto& it : circuit.wires) {
        const auto& accumulator = new SAIFAccumulator();
        it->accumulator = accumulator;
        accumulators.push_back(accumulator);
    }
}

void SAIFResult::write(char *path) {

}
