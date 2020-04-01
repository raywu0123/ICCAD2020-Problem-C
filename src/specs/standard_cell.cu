#include <iostream>
#include <string>
#include "standard_cell.h"
#include "../constants.h"

using namespace std;


STD_CELL_DECLARE_TYPE StandardCell::get_pin_type(const string& pin_name) {
    for (const auto& type_args : declares) {
        for (const auto& arg_name : type_args.second) {
            if (arg_name == pin_name) {
                return type_args.first;
            }
        }
    }
    return STD_CELL_SUPPLY0;
}

unordered_map<string, shared_ptr<StandardCell>> StandardCellLibrary::cells = unordered_map<string, shared_ptr<StandardCell>>();

shared_ptr<StandardCell> StandardCellLibrary::get_cell_by_type(const string &type) {
    const auto it = cells.find(type);
    if (it == cells.end()) return nullptr;
    return it->second;
}

STD_CELL_DECLARE_TYPE StandardCellLibrary::get_pin_type(const string &cell_type, const string &pin_name) {
    const auto& cell = get_cell_by_type(cell_type);
    return cell->get_pin_type(pin_name);
}
