#include <iostream>
#include "src/circuit_model/circuit.h"
#include "src/input_waveforms.h"
#include "src/utils.h"
#include "src/simulation_result.h"
#include "simulator/simulator.h"
#include "simulator/memory_manager.h"
#include "simulator/module_registry.h"


using namespace std;


void print_usage() {
    cout << "Usage: GPUSimulator.cu.py "
            "<intermediate_representation.file> "
            "<input.vcd> <SAIF_or_VCD_flag> "
            "[SAIF_or_output_VCD.saif.vcd]" << endl;
}


bool arguments_valid(int argc, char* argv[1]) {
    if (argc != 5) {
        cerr << "Wrong number of arguments" << endl;
        print_usage();
        return false;
    }
    string output_flag = string(argv[3]);
    if (output_flag != "SAIF" and output_flag != "VCD") {
        cerr << "The third argument should be either 'SAIF' or 'VCD'" << endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (not arguments_valid(argc, argv))
        return -1;

    char* inter_repr_file = argv[1];
    char* input_vcd_file = argv[2];
    string output_flag = string(argv[3]);
    char* output_file = argv[4];

    InputWaveforms input_waveforms(input_vcd_file);
    input_waveforms.summary();

    ifstream fin = ifstream(inter_repr_file);
    ModuleRegistry module_registry;
    module_registry.read_file(fin);
    module_registry.summary();

    BusManager bus_manager;
    Circuit circuit(module_registry);
    circuit.read_file(fin, input_waveforms.timescale, bus_manager, output_flag);
    input_waveforms.get_input_wires(circuit);
    circuit.summary();

    SimulationResult* simulation_result;
    if (output_flag == "SAIF") {
        simulation_result = new SAIFResult(
            circuit.wires,
            input_waveforms.scopes,
            input_waveforms.timescale_pair
        );
    } else if (output_flag == "VCD") {
        simulation_result = new VCDResult(
            circuit.wires,
            input_waveforms.scopes,
            input_waveforms.timescale_pair,
            bus_manager
        );
    }


    MemoryManager::init();
    Simulator simulator(circuit, input_waveforms, simulation_result);
    simulator.run();

    simulation_result->write(output_file);
}
