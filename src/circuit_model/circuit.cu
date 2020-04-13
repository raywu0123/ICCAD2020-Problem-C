#include <iostream>

#include "circuit.h"
#include "constants.h"

using namespace std;

extern double get_timescale(int, const string&);


void Circuit::summary() const {
    cout << "Summary of Circuit" << endl;
    cout << "Num cells: " << cells.size() << endl;
    cout << "Num wires: " << wires.size() << endl;
    cout << "Num schedule layers: " << cell_schedule.size() << endl;
    cout << endl;
}

Wire* Circuit::get_wire(const Wirekey& wirekey) const {
    const auto& it = wires.find(wirekey);
    if (it == wires.end())
        throw runtime_error("Wire " + wirekey.first + " index " + to_string(wirekey.second) + " not found.");
    return it->second;
}

Cell* Circuit::get_cell(const string& name) const {
    const auto& it = cells.find(name);
    if (it == cells.end())
        throw runtime_error("Cell " + name + " not found");
    return it->second;
}

void Circuit::read_file(ifstream &fin, double input_timescale) {
    fin >> design_name;
    read_wires(fin);
    read_assigns(fin);
    read_cells(fin);
    read_schedules(fin);
    read_sdf(fin, input_timescale);
}

void Circuit::read_wires(ifstream& fin) {
    unsigned int num_wires;
    fin >> num_wires;
    wires.reserve(num_wires);
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
    unsigned int num_cells;
    fin >> num_cells;
    cells.reserve(num_cells);

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
        unsigned int num_cell, num_alloc_wire, num_free_wire;
        fin >> num_cell;
        vector<Cell*> cell_ids;
        cell_ids.reserve(num_cell);
        for (int j = 0; j < num_cell; j++) {
            string cell_id;
            fin >> cell_id;
            cell_ids.emplace_back(get_cell(cell_id));
        }
        cell_schedule.emplace_back(cell_ids);

        fin >> num_alloc_wire;
        vector<Wire*> alloc_wirekeys;
        alloc_wirekeys.reserve(num_alloc_wire);
        for (int j = 0; j < num_alloc_wire; j++) {
            string wire_name;
            int wire_index;
            fin >> wire_name >> wire_index;
            alloc_wirekeys.emplace_back(get_wire(make_pair(wire_name, wire_index)));
        }
        wire_alloc_schedule.emplace_back(alloc_wirekeys);

        fin >> num_free_wire;
        vector<Wire*> free_wirekeys;
        free_wirekeys.reserve(num_free_wire);
        for (int j = 0; j < num_free_wire; j++) {
            string wire_name;
            int wire_index;
            fin >> wire_name >> wire_index;
            free_wirekeys.emplace_back(get_wire(make_pair(wire_name, wire_index)));
        }
        wire_free_schedule.emplace_back(free_wirekeys);
    }
}

void Circuit::register_01_wires() {
    wires.emplace(make_pair("1'b1", 0), new ConstantWire('1'));
    wires.emplace(make_pair("1'b0", 0), new ConstantWire('0'));
}

void Circuit::read_sdf(ifstream &fin, double input_timescale) const {
    string s, timescale_unit;
    int timescale_num;
    fin >> s >> timescale_num >> timescale_unit;
    double sdf_timescale = get_timescale(timescale_num, timescale_unit);

    unsigned int num_cells;
    fin >> num_cells;
    for (int i_cell = 0; i_cell < num_cells; i_cell++) {
        string type, name;
        unsigned int num_paths;
        fin >> type >> name >> num_paths;
        vector<SDFPath> paths;
        paths.reserve(num_paths);
        for(int i_path = 0; i_path < num_paths; i_path++) {
            SDFPath path;
            double sdf_rising_delay, sdf_falling_delay;
            fin >> path.in >> path.out >> sdf_rising_delay >> sdf_falling_delay;

            path.rising_delay = (int)(sdf_rising_delay * sdf_timescale / input_timescale);
            path.falling_delay = (int)(sdf_falling_delay * sdf_timescale / input_timescale);
            paths.push_back(path);
        }
        bind_sdf_to_cell(name, paths);
    }
}

void Circuit::bind_sdf_to_cell(const string& name, const vector<SDFPath>& paths) const {
    get_cell(name)->set_paths(paths);
}

void Circuit::register_input_wires(const vector<Bucket>& buckets) {
    for(const auto& bucket: buckets)
        input_wires.push_back(get_wire(bucket.wirekey));
}

void Wire::set_input(const vector<Transition>&, int start_index, int size) {

}
