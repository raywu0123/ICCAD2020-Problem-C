#ifndef ICCAD2020_MODULE_REGISTRY_H
#define ICCAD2020_MODULE_REGISTRY_H

#include <string>
#include <vector>

#include "data_structures.h"

using namespace std;


class ModuleRegistry {
public:
    ModuleRegistry();
    void summary() const;

    void register_primitives();
    void register_user_defined_primitive(
        const string& name,
        const vector<string>& table
    );
    void register_module(const string& name, const vector<pair<string, vector<int>>>& submodules);

    GateFnPtr get_gate_fn(const string& name, char*& table) const;


private:

    unordered_map<string, GateFnPtr> name_to_gate{};
    unordered_map<string, char*> name_to_table{};
    unordered_map<string, ModuleSpec> name_to_module_spec{};
};

#endif
