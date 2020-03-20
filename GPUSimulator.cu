#include <iostream>

using namespace std;


__global__ void cuda_hello(){
}

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
        cerr << "The second argument should be either 'SAIF' or 'VCD'" << endl;
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

    cuda_hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
