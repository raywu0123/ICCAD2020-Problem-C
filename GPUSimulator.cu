#include <iostream>
#include "src/circuit_model/circuit.h"
#include "src/vcd_reader.h"
#include "src/utils.h"
#include "src/simulation_result.h"
#include "simulator/simulator.h"
#include "simulator/module_registry.h"


using namespace std;


void print_usage() {
    cout << "| Usage: GPUSimulator.cu.py "
            "<intermediate_representation.file> "
            "<input.vcd> "
            "<dumpon_time> "
            "<dumpoff_time> "
            "[SAIF_or_output_VCD.saif.vcd] "
            "[SAIF_or_VCD_flag]\n";
}


bool arguments_valid(int argc, char* argv[1]) {
    if (argc < 5) {
        cerr << "| Error: Wrong number of arguments" << endl;
        print_usage();
        return false;
    }
    if (atoll(argv[3]) >= atoll(argv[4])) {
        cerr << "| Error: dumpoff_time earlier than dumpon_time" << endl;
        return false;
    }
    if (atoll(argv[3]) < 0) {
        cerr << "| Error: negative dumpon_time" << endl;
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
    Timestamp dumpon_time = atoll(argv[3]);
    Timestamp dumpoff_time = atoll(argv[4]);

    ifstream fin_intermediate = ifstream(inter_repr_file);
    if (!fin_intermediate) throw runtime_error("Bad intermediate file.");
    ModuleRegistry module_registry;
    module_registry.read_file(fin_intermediate);
    module_registry.summary();

    BusManager bus_manager;
    Circuit circuit(module_registry);
    VCDReader vcd_reader(input_vcd_file);
    InputInfo input_info = vcd_reader.read_input_info();
    input_info.summary();

    circuit.read_intermediate_file(fin_intermediate, input_info.timescale, bus_manager);
    fin_intermediate.close();
    vcd_reader.read_input_waveforms(circuit);
    vcd_reader.summary();
    circuit.summary();

    Simulator simulator(circuit);
    simulator.run();
    MemoryManager::finish();

    SimulationResult* simulation_result;
    const auto& referenced_wires = circuit.get_referenced_wires();

    if (argc == 5) return 0;

    char* output_file_name = argv[5];
    string output_flag = argc == 6 ? "SAIF" : string(argv[6]);
    if (output_flag == "SAIF") {
        simulation_result = new SAIFResult(
            referenced_wires,
            input_info.scopes,
            input_info.timescale_pair,
            dumpon_time, dumpoff_time,
            bus_manager
        );
    } else if (output_flag == "VCD") {
        simulation_result = new VCDResult(
            referenced_wires,
            input_info.scopes,
            input_info.timescale_pair,
            dumpon_time, dumpoff_time,
            bus_manager
        );
    } else throw runtime_error("Invalid output flag " + output_flag + "\n");

    simulation_result->write(output_file_name);
    delete simulation_result;
}
