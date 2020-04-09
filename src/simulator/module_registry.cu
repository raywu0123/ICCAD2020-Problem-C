#include <iostream>
#include "module_registry.h"


ModuleRegistry::ModuleRegistry() {
    register_primitives();
}

void ModuleRegistry::summary() const {
    cout << "Module Registry Summary: " << endl;
    cout << "Num gates: " << name_to_gate.size() << endl;
    cout << "Num modules: " << name_to_module.size() << endl;
}

void ModuleRegistry::register_primitives() {
    name_to_gate["and"] = new ANDGate();
    name_to_gate["or"] = new ORGate();
    name_to_gate["not"] = new NOTGate();
    name_to_gate["nor"] = new NORGate();
    name_to_gate["xor"] = new XORGate();
    name_to_gate["xnor"] = new XNORGate();
    name_to_gate["nand"] = new NANDGate();
    name_to_gate["buf"] = new BUFGate();
}

void ModuleRegistry::register_user_defined_primitive(
        const string &name,
        const vector<string> &table,
        int input_size, int output_size) {
    if (name_to_gate.find(name) != name_to_gate.end()) {
        throw runtime_error("Duplicate primitive names: " + name);
    }
    Primitive* p = Primitive(table, input_size, output_size).cuda();
    name_to_gate[name] = p;
}

Gate *ModuleRegistry::get_gate(const string &name) const {
    const auto& it = name_to_gate.find(name);
    if (it == name_to_gate.end()) {
        return nullptr;
    }
    return it->second;
}

void ModuleRegistry::register_module(const string &name, const vector<pair<string, vector<int>>>& submodules) {
    if (name_to_module.find(name) != name_to_module.end()) {
        throw runtime_error("Duplicate module names: " + name);
    }
    vector<pair<Gate*, vector<int>>> ptr_submodules;
    for (const auto& p : submodules) {
        ptr_submodules.emplace_back(get_gate(p.first), p.second);
    }
    name_to_module[name] = new Module(ptr_submodules);
}

Module *ModuleRegistry::get_module(const string &name) const {
    const auto& it = name_to_module.find(name);
    if (it == name_to_module.end()) {
        return nullptr;
    }
    return it->second;
}
