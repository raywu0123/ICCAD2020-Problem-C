#include <iostream>
#include "circuit.h"
#include "constants.h"


using namespace std;

void Circuit::summary() const {
    cout << "Summary of Circuit" << endl;
    cout << "Num cells: " << cells.size() << endl;
    cout << "Num wires: " << wires.size() << endl;
    cout << "Num schedule layers: " << cell_schedule.size() << endl;
    cout << "Num input wires: " << cell_schedule.size() << endl;
}

const Wire* Circuit::get_wire(const Wirekey& wirekey) const {
    const auto& it = wires.find(wirekey);
    if (it == wires.end())
        throw runtime_error("Wire " + wirekey.first + " index " + to_string(wirekey.second) + " not found.");
    return it->second;
}

void Circuit::read_file(ifstream &fin) {
    fin >> design_name;
    read_wires(fin);
    read_assigns(fin);
    read_cells(fin);
    read_schedules(fin);
}

void Circuit::read_wires(ifstream& fin) {
    int num_wires;
    fin >> num_wires;
    for (int i = 0; i < num_wires; i++) {
        string wire_name;
        int wire_index;
        fin >> wire_name >> wire_index;
        wires.emplace(make_pair(wire_name, wire_index), new Wire());
    }
}

void Circuit::read_assigns(ifstream& fin) {
    int num_assigns;
    fin >> num_assigns;
    for (int i = 0; i < num_assigns; i++) {
        string lhs_wire_name, rhs_wire_name;
        int lhs_wire_index, rhs_wire_index;
        fin >> lhs_wire_name >> lhs_wire_index >> rhs_wire_name >> rhs_wire_index;
        Wirekey lhs_wirekey = make_pair(lhs_wire_name, lhs_wire_index);
        Wirekey rhs_wirekey = make_pair(rhs_wire_name, rhs_wire_index);
        const auto& lhs_wire_ptr = get_wire(lhs_wirekey);
        const auto& rhs_wire_ptr = get_wire(rhs_wirekey);

        delete lhs_wire_ptr;
        wires[lhs_wirekey] = rhs_wire_ptr;  // both wirekeys now share the same Wire instance
    }
}

void Circuit::read_cells(ifstream& fin) {
    int num_cells;
    fin >> num_cells;
    for (int i = 0; i < num_cells; i++) {
        string cell_name, cell_type;
        unsigned int num_args;
        fin >> cell_type >> cell_name >> num_args;

        vector<PinSpec> args{num_args};
        for (auto& arg: args)
            fin >> arg.name >> arg.type >> arg.wirekey.first >> arg.wirekey.second;

        cells.emplace(cell_name, create_cell(cell_type, args));
    }
}

Cell* Circuit::create_cell(const string& cell_type, const vector<PinSpec>& pin_specs) {
    const ModuleSpec* module_spec = module_registry.get_module_spec(cell_type);
    const vector<SubmoduleSpec>* submodule_specs = module_registry.get_submodule_specs(cell_type);

    unordered_map<string, const Wire*> cell_wires;
    for (const auto& pin_spec : pin_specs) {
        const auto& wire_ptr = get_wire(pin_spec.wirekey);
        cell_wires[pin_spec.name] = wire_ptr;
    }
    const StdCellDeclare* declare = module_registry.get_module_declare(cell_type);
    const Wire *supply1_wire = get_wire(SUPPLY1_WIREKEY), *supply0_wire = get_wire(SUPPLY0_WIREKEY);
    return new Cell(module_spec, submodule_specs, declare, cell_wires, supply1_wire, supply0_wire);
}

void Circuit::read_schedules(ifstream& fin) {
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
        for (int j = 0; j < num_alloc_wire; j++) {
            string wire_name;
            int wire_index;
            fin >> wire_name >> wire_index;
            alloc_wirekeys.emplace_back(wire_name, wire_index);
        }
        wire_alloc_schedule.emplace_back(alloc_wirekeys);

        fin >> num_free_wire;
        free_wirekeys.reserve(num_alloc_wire);
        for (int j = 0; j < num_free_wire; j++) {
            string wire_name;
            int wire_index;
            fin >> wire_name >> wire_index;
            free_wirekeys.emplace_back(wire_name, wire_index);
        }
        wire_free_schedule.emplace_back(free_wirekeys);
    }
}

void Circuit::register_01_wires() {
    wires.emplace(make_pair("1'b1", 0), new ConstantWire('1'));
    wires.emplace(make_pair("1'b0", 0), new ConstantWire('0'));
}

void Circuit::register_input_wire(const Wirekey& wirekey) {
    if (input_wires.find(wirekey) != input_wires.end())
        throw runtime_error("Duplicated input wire key: " + wirekey.first + "\n");
    input_wires.insert(wirekey);
}
