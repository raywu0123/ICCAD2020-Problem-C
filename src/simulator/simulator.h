#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include <vector>
#include <queue>
#include <mutex>

#include "circuit_model/circuit.h"
#include "job.h"
#include "simulation_result.h"


class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

    static void worker(std::mutex& m, std::queue<Job*>& job_queue);

    Circuit& circuit;
};

#endif
