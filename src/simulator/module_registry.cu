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
            unsigned int arg_index;
            fin >> arg_index;
            submodule_spec.args.push_back(arg_index);
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
        unsigned int arg_index;
        string s;
        fin >> s >> num_args;
        for (int i = 0; i < num_args; i++){
            fin >> arg_index;
            arg_bucket.push_back(arg_index);
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

//    temporary
//    TODO move module_spec to constant memory
    if (submodules.empty())
        throw runtime_error("Empty module " + name + "\n");
    vector<unsigned int> data_schedule_indices;
    for (const auto& submodule_spec : submodules) {
        data_schedule_indices.insert(
            data_schedule_indices.end(),
            submodule_spec.args.begin(),
            submodule_spec.args.end()
        );
    }

    vector<GateFnPtr> gate_schedule;
    vector<char*> tables;
    vector<unsigned int> table_row_nums, num_inputs, num_outputs;
    unsigned int schedule_size = submodules.size();
    for (int i = 0; i < schedule_size; i++) {
        char* table;
        unsigned int table_row_num;
        gate_schedule.push_back(get_gate_fn(submodules[i].type, table, table_row_num));
        tables.push_back(table);
        table_row_nums.push_back(table_row_num);
        num_outputs.push_back(1);
        num_inputs.push_back(submodules[i].args.size() - 1);
    }
    ModuleSpec device_module_spec_{};
    device_module_spec_.schedule_size = schedule_size;
    device_module_spec_.data_schedule_size = data_schedule_indices.size();
    cudaMalloc((void**) &device_module_spec_.data_schedule_indices, sizeof(unsigned int) * data_schedule_indices.size());
    cudaMemcpy(device_module_spec_.data_schedule_indices, data_schedule_indices.data(), sizeof(unsigned int) * data_schedule_indices.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.gate_schedule, sizeof(GateFnPtr) * schedule_size);
    cudaMemcpy(device_module_spec_.gate_schedule, gate_schedule.data(), sizeof(GateFnPtr) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.tables, sizeof(char*) * schedule_size);
    cudaMemcpy(device_module_spec_.tables, tables.data(), sizeof(char*) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.table_row_num, sizeof(unsigned int) * schedule_size);
    cudaMemcpy(device_module_spec_.table_row_num, table_row_nums.data(), sizeof(unsigned int) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.num_inputs, sizeof(unsigned int) * schedule_size);
    cudaMemcpy(device_module_spec_.num_inputs, num_inputs.data(), sizeof(unsigned int) * schedule_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &device_module_spec_.num_outputs, sizeof(unsigned int) * schedule_size);
    cudaMemcpy(device_module_spec_.num_outputs, num_outputs.data(), sizeof(unsigned int) * schedule_size, cudaMemcpyHostToDevice);

    ModuleSpec* device_module_spec;
    cudaMalloc((void**) &device_module_spec, sizeof(ModuleSpec));
    cudaMemcpy(device_module_spec, &device_module_spec_, sizeof(ModuleSpec), cudaMemcpyHostToDevice);
    name_to_module_spec[name] = device_module_spec;
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
