#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "circuit_model/circuit.h"
#include "simulation_result.h"
#include "data_structures.h"


struct ResourceBuffer {
    std::vector<const ModuleSpec*> module_specs;
    std::vector<const Transition*> data_schedule;
    std::vector<unsigned int> data_schedule_offsets;
    std::vector<unsigned int> capacities;

    void clear() {
        module_specs.clear();
        data_schedule.clear();
        data_schedule_offsets.clear();
        capacities.clear();
    }

    void push_back(const CellResource& resource) {
        module_specs.push_back(resource.module_spec);
        data_schedule.insert(data_schedule.end(), resource.data_schedule.begin(), resource.data_schedule.end());
        data_schedule_offsets.push_back(data_schedule.size());
        capacities.insert(capacities.end(), resource.capacities.begin(), resource.capacities.end());
    }

    int size() const { return module_specs.size(); }
};

class Simulator {
public:
    Simulator(
        Circuit& c,
        InputWaveforms& iw,
        SimulationResult* sr
    ): circuit(c), simulation_result(sr), input_waveforms(iw) {};

    void run();
    void simulate_batch_stimuli(std::vector<unsigned long>& stimuli_indices);
    void set_input(std::vector<unsigned long>& stimuli_indices) const;

    BatchResource get_batch_data();

    Circuit& circuit;
    SimulationResult* simulation_result;
    InputWaveforms& input_waveforms;
    ResourceBuffer resource_buffer;
};

#endif
