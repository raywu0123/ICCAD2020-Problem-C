#include <iostream>
#include "src/circuit_model/circuit.h"
#include "src/input_waveforms.h"
#include "simulator/module_registry.h"
#include "src/simulation_result.h"
#include "simulator/simulator.h"
#include "src/utils.h"


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
    string argv3(argv[3]);
    if (argv3 != string("SAIF") and argv3 != "VCD") {
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
    char* saif_or_vcd_flag = argv[3];
    char* output_file = argv[4];

    InputWaveforms input_waveforms(input_vcd_file);
    input_waveforms.summary();

    ifstream fin = ifstream(inter_repr_file);
    ModuleRegistry module_registry;
    module_registry.read_file(fin);
    module_registry.summary();

    Circuit circuit(module_registry);
    circuit.read_file(fin, input_waveforms.timescale);
    circuit.register_input_wires(input_waveforms.token_to_wire);
    circuit.summary();

    SimulationResult simulation_result;
    Simulator simulator(circuit, input_waveforms, simulation_result);
    simulator.run();

    simulation_result.write(output_file, saif_or_vcd_flag);
}
