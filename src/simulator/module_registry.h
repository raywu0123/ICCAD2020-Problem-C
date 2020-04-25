#ifndef ICCAD2020_MODULE_REGISTRY_H
#define ICCAD2020_MODULE_REGISTRY_H

#include <string>
#include <vector>
#include <unordered_map>

#include "data_structures.h"

struct StdCellDeclare {
    std::vector<std::vector<unsigned int>> buckets{5};
};

struct Table {
    char* table = nullptr;
    unsigned int num_rows = 0;
};


class ModuleRegistry {
public:
    ModuleRegistry();

    void read_file(std::ifstream&);
    void summary() const;
    GateFnPtr get_gate_fn(const std::string& name, char*& table, unsigned int& table_row_num) const;
    const ModuleSpec* get_module_spec(const std::string& cell_type) const;
    const std::vector<SubmoduleSpec>* get_submodule_specs(const std::string& cell_type) const;
    const StdCellDeclare* get_module_declare(const std::string& cell_type) const;

private:
    void read_vlib_primitive(std::ifstream&);
    void read_vlib_module(std::ifstream&);
    static std::string read_vlib_common(std::ifstream&, StdCellDeclare&);
    static void read_vlib_table(std::ifstream&, std::vector<std::string>&);

    void register_primitives();
    void register_user_defined_primitive(
        const std::string& name,
        const std::vector<std::string>& table,
        const StdCellDeclare& declares
    );
    void register_module(const std::string& name, const std::vector<SubmoduleSpec>& submodules, const StdCellDeclare& declares);

    std::unordered_map<std::string, Table> name_to_table{};
    std::unordered_map<std::string, ModuleSpec*> name_to_module_spec{};  // to be transferred to device
    std::unordered_map<std::string, StdCellDeclare> name_to_declares{};
    std::unordered_map<std::string, std::vector<SubmoduleSpec>> name_to_submodule_specs{};
    std::unordered_map<std::string, GateFnPtr> name_to_gate{};
};

#endif
