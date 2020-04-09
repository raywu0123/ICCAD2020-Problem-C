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
        const vector<string>& table,
        int input_size, int output_size
    );
    void register_module(const string& name, const vector<pair<string, vector<int>>>& submodules);

    Gate* get_gate(const string& name) const;
    Module* get_module(const string& name) const;

    char** get_module_data() const {
        return nullptr;
    }

private:

    unordered_map<string, Gate*> name_to_gate{};
    unordered_map<string, Module*> name_to_module{};
};

#endif
