#include <iostream>
#include <fstream>
#include "module_registry.h"
#include "builtin_gates.h"

using namespace std;


ModuleRegistry::ModuleRegistry() {
    register_primitives();
}
//TODO destructor: free module_spec and char table

void ModuleRegistry::summary() const {
    cout << "Module Registry Summary: " << endl;
    cout << "Num gates: " << name_to_gate.size() << endl;
    cout << "Num UDP: " << name_to_table.size() << endl;
    cout << "Num modules: " << name_to_module_spec.size() << endl;
    cout << endl;
}


void ModuleRegistry::read_file(ifstream& fin) {
    int num_primitive_cells, num_modules;
    fin >> num_primitive_cells >> num_modules;

    for (int i = 0; i < num_primitive_cells; i++) {
        read_vlib_primitive(fin);
    }
    for (int i = 0; i < num_modules; i++) {
        read_vlib_module(fin);
    }
}

void ModuleRegistry::read_vlib_primitive(ifstream& fin) {
    StdCellDeclare declares;
    string name = read_vlib_common(fin, declares);
    vector<string> table;
    read_vlib_table(fin, table);

    register_user_defined_primitive(name, table, declares);
}

void ModuleRegistry::read_vlib_table(ifstream& fin, vector<string>& table) {
    int num_rows;
    fin >> num_rows;
    string row;
    for (int i = 0; i < num_rows; i++) {
        fin >> row;
        table.push_back(row);
    }
}

void ModuleRegistry::read_vlib_module(ifstream& fin) {
    StdCellDeclare declares;
    string name = read_vlib_common(fin, declares);
    int num_submodules;

    fin >> num_submodules;
    vector<SubmoduleSpec> submodule_specs;
    for (int i = 0; i < num_submodules; i++) {
        string s;
        int num_args;
        SubmoduleSpec submodule_spec;
        fin >> s >> submodule_spec.type >> submodule_spec.name >> num_args;
        if (num_args < 2)
            throw runtime_error("Less than two args to submodule " + submodule_spec.name + " in " + name + '\n');

        for (int i_arg = 0; i_arg < num_args; i_arg++) {
            string arg;
            fin >> arg;
            submodule_spec.args.push_back(arg);
        }
        submodule_specs.push_back(submodule_spec);
    }
    register_module(name, submodule_specs, declares);
}

string ModuleRegistry::read_vlib_common(ifstream& fin, StdCellDeclare& declares) {
    string name;
    fin >> name;
    for (auto& arg_bucket : declares.buckets) {
        int num_args;
        string s;
        fin >> s >> num_args;
        for (int i = 0; i < num_args; i++) {
            fin >> s;
            arg_bucket.push_back(s);
        }
    }
    return name;
}

extern __device__ GateFnPtr and_gate_fn_ptr;
extern __device__ GateFnPtr or_gate_fn_ptr;
extern __device__ GateFnPtr xor_gate_fn_ptr;
extern __device__ GateFnPtr nand_gate_fn_ptr;
extern __device__ GateFnPtr nor_gate_fn_ptr;
extern __device__ GateFnPtr xnor_gate_fn_ptr;
extern __device__ GateFnPtr not_gate_fn_ptr;
extern __device__ GateFnPtr buf_gate_fn_ptr;
void ModuleRegistry::register_primitives() {
    GateFnPtr host_and_gate_fn_ptr;
    GateFnPtr host_or_gate_fn_ptr;
    GateFnPtr host_xor_gate_fn_ptr;
    GateFnPtr host_nand_gate_fn_ptr;
    GateFnPtr host_nor_gate_fn_ptr;
    GateFnPtr host_xnor_gate_fn_ptr;
    GateFnPtr host_not_gate_fn_ptr;
    GateFnPtr host_buf_gate_fn_ptr;
    cudaMemcpyFromSymbol(&host_and_gate_fn_ptr, and_gate_fn_ptr, sizeof(GateFnPtr));
    cudaMemcpyFromSymbol(&host_or_gate_fn_ptr, or_gate_fn_ptr, sizeof(GateFnPtr));
    cudaMemcpyFromSymbol(&host_xor_gate_fn_ptr, xor_gate_fn_ptr, sizeof(GateFnPtr));
    cudaMemcpyFromSymbol(&host_nand_gate_fn_ptr, nand_gate_fn_ptr, sizeof(GateFnPtr));
    cudaMemcpyFromSymbol(&host_nor_gate_fn_ptr, nor_gate_fn_ptr, sizeof(GateFnPtr));
    cudaMemcpyFromSymbol(&host_xnor_gate_fn_ptr, xnor_gate_fn_ptr, sizeof(GateFnPtr));
    cudaMemcpyFromSymbol(&host_not_gate_fn_ptr, not_gate_fn_ptr, sizeof(GateFnPtr));
    cudaMemcpyFromSymbol(&host_buf_gate_fn_ptr, buf_gate_fn_ptr, sizeof(GateFnPtr));
    name_to_gate["and"] = host_and_gate_fn_ptr;
    name_to_gate["or"] = host_or_gate_fn_ptr;
    name_to_gate["xor"] = host_xor_gate_fn_ptr;
    name_to_gate["nand"] = host_nand_gate_fn_ptr;
    name_to_gate["nor"] = host_nor_gate_fn_ptr;
    name_to_gate["xnor"] = host_xnor_gate_fn_ptr;
    name_to_gate["not"] = host_not_gate_fn_ptr;
    name_to_gate["buf"] = host_buf_gate_fn_ptr;
}

void ModuleRegistry::register_user_defined_primitive(
    const string &name,
    const vector<string>& table,
    const StdCellDeclare& declares
) {
    if (name_to_gate.find(name) != name_to_gate.end()) {
        throw runtime_error("Primitive names shadows gate name: " + name + "\n");
    }
    if (table.empty()) {
        throw runtime_error("Emtpy table for UDP: " + name + "\n");
    }
    if(name_to_declares.find(name) != name_to_declares.end()) {
        throw runtime_error("Duplicate modules: " + name + "\n");
    }
    name_to_declares[name] = declares;

    int row_size = table[0].size();
    Table table_struct;
    table_struct.num_rows = table.size();
    table_struct.table = new char[table_struct.num_rows * row_size]; // temporary
//    TODO move char_table to constant memory
    for(int i = 0; i < table_struct.num_rows; i++) {
        for (int j = 0; j < row_size; j++) {
            table_struct.table[i * row_size + j] = table[i][j];
        }
    }
    char* device_char_table;
    cudaMalloc((void**) &device_char_table, table_struct.num_rows * row_size);
    cudaMemcpy(device_char_table, table_struct.table, sizeof(table_struct.num_rows) * row_size, cudaMemcpyHostToDevice);
    delete[] table_struct.table;
    table_struct.table = device_char_table;
    name_to_table[name] = table_struct;
}

GateFnPtr ModuleRegistry::get_gate_fn(const string &name, char*& table, unsigned int& table_row_num) const {
    const auto& gate_it = name_to_gate.find(name);
    if (gate_it != name_to_gate.end()) return gate_it->second;
    const auto& table_it = name_to_table.find(name);
    if (table_it != name_to_table.end()) {
        table = table_it->second.table;
        table_row_num = table_it->second.num_rows;
        return nullptr;
    }

    throw runtime_error("Gate " + name + " not found.\n");
}

void ModuleRegistry::register_module(
    const string& name,
    const vector<SubmoduleSpec>& submodules,
    const StdCellDeclare& declares
) {
    if(name_to_module_spec.find(name) != name_to_module_spec.end()) {
        throw runtime_error("Duplicate modules: " + name + "\n");
    }
    if(name_to_submodule_specs.find(name) != name_to_submodule_specs.end()) {
        throw runtime_error("Duplicate modules: " + name + "\n");
    }
    if(name_to_declares.find(name) != name_to_declares.end()) {
        throw runtime_error("Duplicate modules: " + name + "\n");
    }
    name_to_submodule_specs[name] = submodules;
    name_to_declares[name] = declares;

    ModuleSpec module_spec{};
//    TODO move module_spec to constant memory
    module_spec.schedule_size = submodules.size();
    if (submodules.empty())
        throw runtime_error("Empty module " + name + "\n");
    unsigned int schedule_size = submodules.size();
//    temporary
    module_spec.gate_schedule = new GateFnPtr[schedule_size];
    module_spec.tables = new char*[submodules.size()];
    module_spec.table_row_num = new unsigned int[schedule_size];
    module_spec.num_inputs = new unsigned int[schedule_size];
    module_spec.num_outputs = new unsigned int[schedule_size];
    for (int i = 0; i < schedule_size; i++) {
        module_spec.gate_schedule[i] = get_gate_fn(
            submodules[i].type,
            module_spec.tables[i],
            module_spec.table_row_num[i]
        );
        module_spec.num_outputs[i] = 1;
        module_spec.num_inputs[i] = submodules[i].args.size() - 1;
    }
    ModuleSpec device_module_spec_{};
    device_module_spec_.schedule_size = schedule_size;
    cudaMalloc((void**) &device_module_spec_.gate_schedule, sizeof(GateFnPtr) * schedule_size);
    cudaMemcpy(device_module_spec_.gate_schedule, module_spec.gate_schedule, sizeof(GateFnPtr) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.tables, sizeof(char*) * schedule_size);
    cudaMemcpy(device_module_spec_.tables, module_spec.tables, sizeof(char*) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.table_row_num, sizeof(unsigned int) * schedule_size);
    cudaMemcpy(device_module_spec_.table_row_num, module_spec.table_row_num, sizeof(unsigned int) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.num_inputs, sizeof(unsigned int) * schedule_size);
    cudaMemcpy(device_module_spec_.num_inputs, module_spec.num_inputs, sizeof(unsigned int) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.num_outputs, sizeof(unsigned int) * schedule_size);
    cudaMemcpy(device_module_spec_.num_outputs, module_spec.num_outputs, sizeof(unsigned int) * schedule_size, cudaMemcpyHostToDevice);

    ModuleSpec* device_module_spec;
    cudaMalloc((void**) &device_module_spec, sizeof(ModuleSpec));
    cudaMemcpy(device_module_spec, &device_module_spec_, sizeof(ModuleSpec), cudaMemcpyHostToDevice);
    name_to_module_spec[name] = device_module_spec;

    delete[] module_spec.gate_schedule;
    delete[] module_spec.tables;
    delete[] module_spec.table_row_num;
    delete[] module_spec.num_inputs;
    delete[] module_spec.num_outputs;
}

const ModuleSpec* ModuleRegistry::get_module_spec(const string &cell_type) const {
    const auto& it = name_to_module_spec.find(cell_type);
    if (it == name_to_module_spec.end())
        throw runtime_error("ModuleSpec for type " + cell_type + " not found.");
    return it->second;
}

const vector<SubmoduleSpec>* ModuleRegistry::get_submodule_specs(const string &cell_type) const {
    const auto& it = name_to_submodule_specs.find(cell_type);
    if (it == name_to_submodule_specs.end())
        throw runtime_error("SubmoduleSpecs for type " + cell_type + " not found.");
    return &it->second;
}

const StdCellDeclare* ModuleRegistry::get_module_declare(const string &cell_type) const {
    const auto& it = name_to_declares.find(cell_type);
    if (it == name_to_declares.end())
        throw runtime_error("Declares for type " + cell_type + " not found.");
    return &it->second;
}
