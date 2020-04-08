#ifndef ICCAD2020_CIRCUIT_H
#define ICCAD2020_CIRCUIT_H

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "constants.h"

using namespace std;

class Wire {

};

class Cell {

};


class Circuit {
public:
    void read_file(ifstream& fin);
    void summary() const;

    string design_name;

    unordered_map<Wirekey, Wire, pair_hash> wires;
    vector<pair<Wirekey, Wirekey>> assigns;
    unordered_map<string, Cell> cells;

    vector<vector<string>> cell_schedule;
    vector<vector<Wirekey>> wire_alloc_schedule;
    vector<vector<Wirekey>> wire_free_schedule;
};

#endif //ICCAD2020_CIRCUIT_H