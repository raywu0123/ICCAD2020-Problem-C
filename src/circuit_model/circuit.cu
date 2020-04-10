#include <iostream>
#include "circuit.h"
#include "constants.h"


using namespace std;


void Circuit::read_file(ifstream &fin) {
    fin >> design_name;

    int num_wires;
    fin >> num_wires;
    for (int i = 0; i < num_wires; i++) {
        string wire_name;
        int wire_index;
        fin >> wire_name >> wire_index;
        wires.emplace(make_pair(wire_name, wire_index), Wire());
    }

    int num_assigns;
    fin >> num_assigns;
    for (int i = 0; i < num_assigns; i++) {
        string lhs_wire_name, rhs_wire_name;
        int lhs_wire_index, rhs_wire_index;
        fin >> lhs_wire_name >> lhs_wire_index >> rhs_wire_name >> rhs_wire_index;
        assigns.emplace_back(
            make_pair(lhs_wire_name, lhs_wire_index),
            make_pair(rhs_wire_name, rhs_wire_index)
        );
    }

    int num_cells;
    fin >> num_cells;
    for (int i = 0; i < num_cells; i++) {
        string cell_name, cell_type;
        int num_args;
        fin >> cell_type >> cell_name >> num_args;
        vector<pair<string, Wirekey>> args(num_args);
        for (int j = 0; j < num_args; j++) {
            string pin_name, wire_name;
            char pin_type;
            int wire_index;
            fin >> pin_name >> pin_type >> wire_name >> wire_index;
            args.emplace_back(pin_name, make_pair(wire_name, wire_index));
        }
        cells.emplace(cell_name, Cell());
    }

    int num_schedule_layers;
    fin >> num_schedule_layers;
    cell_schedule.reserve(num_schedule_layers);
    wire_alloc_schedule.reserve(num_schedule_layers);
    wire_free_schedule.reserve(num_schedule_layers);
    for (int i = 0; i < num_schedule_layers; i++) {
        int num_cell, num_alloc_wire, num_free_wire;
        vector<string> cell_ids;
        fin >> num_cell;
        for (int j = 0; j < num_cell; j++) {
            string cell_id;
            fin >> cell_id;
            cell_ids.emplace_back(cell_id);
        }
        cell_schedule.emplace_back(cell_ids);

        fin >> num_alloc_wire;
        vector<Wirekey> alloc_wirekeys, free_wirekeys;
        alloc_wirekeys.reserve(num_alloc_wire);
        for (int j = 0; j < num_cell; j++) {
            string wire_name;
            int wire_index;
            fin >> wire_name >> wire_index;
            alloc_wirekeys.emplace_back(wire_name, wire_index);
        }
        wire_alloc_schedule.emplace_back(alloc_wirekeys);

        fin >> num_free_wire;
        free_wirekeys.reserve(num_alloc_wire);
        for (int j = 0; j < num_cell; j++) {
            string wire_name;
            int wire_index;
            fin >> wire_name >> wire_index;
            free_wirekeys.emplace_back(wire_name, wire_index);
        }
        wire_free_schedule.emplace_back(free_wirekeys);
    }
}

void Circuit::summary() const {
    cout << "Summary of Circuit" << endl;
    cout << "Num cells: " << cells.size() << endl;
    cout << "Num wires: " << wires.size() << endl;
    cout << "Num schedule layers: " << cell_schedule.size() << endl;
    cout << "Num input wires: " << cell_schedule.size() << endl;
}

void Circuit::register_input_wire(const Wirekey& wirekey) {
    if (input_wires.find(wirekey) != input_wires.end())
        throw runtime_error("Duplicated input wire key: " + wirekey.first + "\n");
    input_wires.insert(wirekey);
}
