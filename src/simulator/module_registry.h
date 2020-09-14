#ifndef ICCAD2020_MODULE_REGISTRY_H
#define ICCAD2020_MODULE_REGISTRY_H

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

#include "data_structures.h"

struct StdCellDeclare {
    NUM_ARG_TYPE num_input, num_output;
};

class ModuleRegistry {
public:
    void read_file(std::ifstream&);
    void summary() const;
    const ModuleSpec* get_module_spec(const std::string& cell_type) const;
    const StdCellDeclare* get_module_declare(const std::string& cell_type) const;

private:
    void read_vlib_module(std::ifstream&);
    void register_module(const std::string& name, const StdCellDeclare& declares, const std::vector<std::string>& table);

    std::unordered_map<std::string, ModuleSpec*> name_to_module_spec{};  // to be transferred to device
    std::unordered_map<std::string, StdCellDeclare> name_to_declares{};
};

#endif
