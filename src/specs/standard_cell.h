#ifndef ICCAD2020_STANDARD_CELL_H
#define ICCAD2020_STANDARD_CELL_H

#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>

#include "../constants.h"

using namespace std;

class StandardCell {
public:
    explicit StandardCell(
        vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>>& declares
    ):  declares(std::move(declares)) {};

    STD_CELL_DECLARE_TYPE get_pin_type(const string& pin_name);

    vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>> declares;
};

class PrimitiveCell : public StandardCell {
public:
    PrimitiveCell(
        vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>>& declares,
        vector<string>& table
    ):  StandardCell(declares){};
};

class Module : public StandardCell {
public:
    Module(
        vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>>& declares,
        const vector<pair<string, vector<string>>>& submodules
    ):  StandardCell(declares) {};
};


class StandardCellLibrary {
public:
    static shared_ptr<StandardCell> get_cell_by_type(const string& type);
    static STD_CELL_DECLARE_TYPE get_pin_type(const string& cell_type, const string& pin_name);
    static unordered_map<string, shared_ptr<StandardCell>> cells;
};

#endif //ICCAD2020_STANDARD_CELL_H
