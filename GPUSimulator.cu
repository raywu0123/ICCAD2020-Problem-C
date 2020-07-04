#include <iostream>
#include "src/circuit_model/circuit.h"
#include "src/vcd_reader.h"
#include "src/utils.h"
#include "src/simulation_result.h"
#include "simulator/simulator.h"
#include "simulator/memory_manager.h"
#include "simulator/module_registry.h"


using namespace std;


void print_usage() {
    cout << "| Usage: GPUSimulator.cu.py "
            "<intermediate_representation.file> "
            "<input.vcd> <SAIF_or_VCD_flag> "
            "[SAIF_or_output_VCD.saif.vcd]" << endl;
}


bool arguments_valid(int argc, char* argv[1]) {
    if (argc != 5) {
        cerr << "| Error: Wrong number of arguments" << endl;
        print_usage();
        return false;
    }
    string output_flag = string(argv[3]);
    if (output_flag != "SAIF" and output_flag != "VCD") {
        cerr << "| Error: The third argument should be either 'SAIF' or 'VCD'" << endl;
        return false;
    }
    return true;
}

void check_cuda_device() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) cerr << "| Error: " << cudaGetErrorString(err) << endl;
}

int main(int argc, char* argv[]) {
    check_cuda_device();

    if (not arguments_valid(argc, argv))
        return -1;

    char* inter_repr_file = argv[1];
    char* input_vcd_file = argv[2];
    string output_flag = string(argv[3]);
    char* output_file = argv[4];

    ifstream fin_intermediate = ifstream(inter_repr_file);
    ModuleRegistry module_registry;
    module_registry.read_file(fin_intermediate);
    module_registry.summary();

    BusManager bus_manager;
    Circuit circuit(module_registry);
    VCDReader vcd_reader(input_vcd_file);
    InputInfo input_info = vcd_reader.read_input_info();
    input_info.summary();

    circuit.read_intermediate_file(fin_intermediate, input_info.timescale, bus_manager);
    vcd_reader.read_input_waveforms(circuit);
    vcd_reader.summary();
    circuit.summary();

    MemoryManager::init();
    Simulator simulator(circuit);
    simulator.run();

    SimulationResult* simulation_result;
    if (output_flag == "SAIF") {
        simulation_result = new SAIFResult(
            circuit.wires,
            input_info.scopes,
            input_info.timescale_pair
        );
    } else if (output_flag == "VCD") {
        simulation_result = new VCDResult(
            circuit.wires,
            input_info.scopes,
            input_info.timescale_pair,
            bus_manager
        );
    }

    simulation_result->write(output_file);
}
