#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "circuit_model/circuit.h"
#include "simulation_result.h"
#include "data_structures.h"


class Simulator {
public:
    Simulator(
        Circuit& c,
        InputWaveforms& iw,
        SimulationResult* sr
    ): circuit(c), simulation_result(sr), input_waveforms(iw) {};

    void run();
    void simulate_batch_stimuli(vector<unsigned long>& stimuli_indices);
    void set_input(vector<unsigned long>& stimuli_indices) const;

    void update_resource(const CellResource&) {
//        TODO
    };
    BatchResource get_batch_data() const {
//        TODO
        return BatchResource{};
    };

    Circuit& circuit;
    SimulationResult* simulation_result;
    InputWaveforms& input_waveforms;


};

#endif
