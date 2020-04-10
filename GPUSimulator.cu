#include <iostream>
#include "src/file_readers/intermediate_file_reader.h"
#include "src/circuit_model/circuit.h"
#include "src/file_readers/vcd_reader.h"
#include "src/input_waveforms.h"
#include "simulator/module_registry.h"
#include "src/simulation_result.h"
#include "simulator/simulator.h"

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

    ifstream fin = ifstream(inter_repr_file);
    ModuleRegistry module_registry;
    module_registry.read_file(fin);

    Circuit circuit(module_registry);
    circuit.read_file(fin);

    TimingSpec timing_spec;
    IntermediateFileReader intermediate_file_reader(timing_spec, fin);

    intermediate_file_reader.read_sdf();
    intermediate_file_reader.summary();
    module_registry.summary();

    InputWaveforms input_waveforms;
    VCDReader vcd_reader(input_waveforms, circuit);
    vcd_reader.read(input_vcd_file);
    vcd_reader.summary();

    circuit.summary();

    SimulationResult simulation_result;
    Simulator simulator(circuit, input_waveforms, simulation_result);
    simulator.run();

    simulation_result.write(output_file, saif_or_vcd_flag);
}
