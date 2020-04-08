#ifndef ICCAD2020_MODULE_REGISTRY_H
#define ICCAD2020_MODULE_REGISTRY_H

#include <string>
#include <vector>

#include "data_structures.h"

using namespace std;


class ModuleRegistry {
public:

    void register_primitive(
            const string& name,
            const vector<string>& table,
            int input_size, int output_size
    ) {
        if (name_to_primitive.find(name) != name_to_primitive.end()) {
            throw runtime_error("Duplicate primitive names: " + name);
        }
        Primitive* p = Primitive(table, input_size, output_size).cuda();
        name_to_primitive[name] = p;
    }

    Primitive* get_primitive(const string& name) const {
        const auto& it = name_to_primitive.find(name);
        if (it == name_to_primitive.end()) {
            return nullptr;
        }
        return it->second;
    }

    void register_module(
            const string& name, const vector<pair<string, vector<int>>>& submodules
    ) {
        if (name_to_module.find(name) != name_to_module.end()) {
            throw runtime_error("Duplicate module names: " + name);
        }
        vector<pair<Primitive*, vector<int>>> ptr_submodules;
        for (const auto& p : submodules) {
            ptr_submodules.emplace_back(get_primitive(p.first), p.second);
        }
        name_to_module[name] = new Module(ptr_submodules);
    }

    Module* get_module(const string& name) const {
        const auto& it = name_to_module.find(name);
        if (it == name_to_module.end()) {
            return nullptr;
        }
        return it->second;
    }
    char** get_module_data() const {
        return nullptr;
    }

private:

    unordered_map<string, Primitive*> name_to_primitive{};
    unordered_map<string, Module*> name_to_module{};
};

#endif
