#ifndef ICCAD2020_MODULE_REGISTRY_H
#define ICCAD2020_MODULE_REGISTRY_H

#include <string>
#include <vector>

#include "data_structures.h"

using namespace std;

struct StdCellDeclare {
    vector<vector<string>> buckets{5};
};


class ModuleRegistry {
public:
    ModuleRegistry();

    void read_file(ifstream&);
    void summary() const;
    GateFnPtr get_gate_fn(const string& name, char*& table) const;
    const ModuleSpec* get_module_spec(const string& cell_type) const;
    const vector<SubmoduleSpec>* get_submodule_specs(const string& cell_type) const;
    const StdCellDeclare* get_module_declare(const string& cell_type) const;

private:
    void read_vlib_primitive(ifstream&);
    void read_vlib_module(ifstream&);
    static string read_vlib_common(ifstream&, StdCellDeclare&);
    static void read_vlib_table(ifstream&, vector<string>&);


    void register_primitives();
    void register_user_defined_primitive(
        const string& name,
        const vector<string>& table,
        const StdCellDeclare& declares
    );
    void register_module(const string& name, const vector<SubmoduleSpec>& submodules, const StdCellDeclare& declares);

    unordered_map<string, GateFnPtr> name_to_gate{};
    unordered_map<string, char*> name_to_table{};
    unordered_map<string, ModuleSpec> name_to_module_spec{};  // to be transferred to device
    unordered_map<string, StdCellDeclare> name_to_declares{};
    unordered_map<string, vector<SubmoduleSpec>> name_to_submodule_specs{};
};

#endif
