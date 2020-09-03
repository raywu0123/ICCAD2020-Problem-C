#include <iostream>
#include "module_registry.h"
#include "utils.h"

using namespace std;

void ModuleRegistry::summary() const {
    cout << "Module Registry Summary: " << endl;
    cout << "Num modules: " << name_to_module_spec.size() << endl;
    cout << endl;
}


void ModuleRegistry::read_file(ifstream& fin) {
    unsigned int num_modules;
    fin >> num_modules;
    for (unsigned int i = 0; i < num_modules; i++) {
        read_vlib_module(fin);
    }
}

void ModuleRegistry::read_vlib_module(ifstream& fin) {
    string name;
    StdCellDeclare declare{};

    unsigned int num_input, num_output;
    fin >> name >> num_input >> num_output;
    declare.num_input = num_input; declare.num_output = num_output;
    unsigned int num_table_rows = pow(4, static_cast<unsigned int>(num_input));

    vector<string> table; table.reserve(num_table_rows);
    for (int i = 0; i < num_table_rows; ++i) {
        string row;
        fin >> row;
        table.push_back(row);
    }

    register_module(name, declare, table);
}

void ModuleRegistry::register_module(
    const string& name,
    const StdCellDeclare& declares,
    const vector<string>& table
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
    name_to_declares[name] = declares;

    // prepare table
    auto table_total_size = table.size() * declares.num_output;
    auto* char_table = new Values[table_total_size];
    const auto& table_row_num = table.size();
    for (int r = 0; r < table_row_num; ++r) {
        for (int o = 0; o < declares.num_output; ++o) {
            char_table[r * declares.num_output + o] = raw_to_enum(table[r][o]);
        }
    }

    // memcpy table to device
    Values* device_char_table;
    cudaMalloc((void**) &device_char_table, sizeof(Values) * table_total_size);
    cudaMemcpy(device_char_table, char_table, sizeof(Values) * table_total_size, cudaMemcpyHostToDevice);

    ModuleSpec device_module_spec_{};
    device_module_spec_.num_input = declares.num_input;
    device_module_spec_.num_output = declares.num_output;
    device_module_spec_.table = device_char_table;

    ModuleSpec* device_module_spec;
    cudaErrorCheck(cudaMalloc((void**) &device_module_spec, sizeof(ModuleSpec)));
    cudaErrorCheck(cudaMemcpy(device_module_spec, &device_module_spec_, sizeof(ModuleSpec), cudaMemcpyHostToDevice));
    name_to_module_spec[name] = device_module_spec;
}

const ModuleSpec* ModuleRegistry::get_module_spec(const string &cell_type) const {
    const auto& it = name_to_module_spec.find(cell_type);
    if (it == name_to_module_spec.end())
        throw runtime_error("ModuleSpec for type " + cell_type + " not found.");
    return it->second;
}

const StdCellDeclare* ModuleRegistry::get_module_declare(const string &cell_type) const {
    const auto& it = name_to_declares.find(cell_type);
    if (it == name_to_declares.end())
        throw runtime_error("Declares for type " + cell_type + " not found.");
    return &it->second;
}
