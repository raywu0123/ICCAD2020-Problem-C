#include <iostream>
#include "module_registry.h"
#include "builtin_gates.h"


ModuleRegistry::ModuleRegistry() {
    register_primitives();
}

void ModuleRegistry::summary() const {
    cout << "Module Registry Summary: " << endl;
    cout << "Num gates: " << name_to_gate.size() << endl;
    cout << "Num UDP: " << name_to_table.size() << endl;
    cout << "Num modules: " << name_to_module_spec.size() << endl;
}

void ModuleRegistry::register_primitives() {
    name_to_gate["and"] = and_gate_fn_ptr;
    name_to_gate["or"] = and_gate_fn_ptr;
    name_to_gate["not"] = and_gate_fn_ptr;
    name_to_gate["xor"] = and_gate_fn_ptr;
    name_to_gate["xnor"] = and_gate_fn_ptr;
    name_to_gate["nand"] = and_gate_fn_ptr;
    name_to_gate["nor"] = and_gate_fn_ptr;
    name_to_gate["buf"] = and_gate_fn_ptr;
}

void ModuleRegistry::register_user_defined_primitive(
    const string &name,
    const vector<string>& table
) {
    if (name_to_gate.find(name) != name_to_gate.end()) {
        throw runtime_error("Duplicate primitive names: " + name + "\n");
    }
    if (table.empty()) {
        throw runtime_error("Emtpy table for UDP: " + name + "\n");
    }
    int row_size = table[0].size();
    int num_rows = table.size();
    char* char_table = new char[num_rows * row_size];
    for(int i = 0; i < num_rows; i++) {
        for (int j = 0; j < row_size; j++) {
            char_table[i * row_size + j] = table[i][j];
        }
    }
    name_to_table[name] = char_table;
}

GateFnPtr ModuleRegistry::get_gate_fn(const string &name, char*& table) const {
    const auto& gate_it = name_to_gate.find(name);
    if (gate_it != name_to_gate.end()) return gate_it->second;

    const auto& table_it = name_to_table.find(name);
    if (table_it != name_to_table.end()) {
        table = table_it->second;
        return PrimitiveGate;
    }

    throw runtime_error("Gate " + name + " not found.\n");
}

void ModuleRegistry::register_module(const string& name, const vector<pair<string, vector<int>>>& submodules) {
    if(name_to_module_spec.find(name) != name_to_module_spec.end()) {
        throw runtime_error("Duplicate modules: " + name + "\n");
    }
    ModuleSpec module_spec{};
    module_spec.schedule_size = submodules.size();
    if (submodules.empty())
        throw runtime_error("Empty module " + name + "\n");

    module_spec.gate_schedule = new GateFnPtr[submodules.size()];
    module_spec.tables = new char*[submodules.size()];
    module_spec.num_inputs = new int[submodules.size()];
    module_spec.num_outputs = new int[submodules.size()];
    for (int i = 0; i < submodules.size(); i++) {
        module_spec.gate_schedule[i] = get_gate_fn(submodules[i].first, module_spec.tables[i]);
        module_spec.num_outputs[i] = 1;
        module_spec.num_inputs[i] = submodules[i].second.size();
    }
    name_to_module_spec[name] = module_spec;
}
