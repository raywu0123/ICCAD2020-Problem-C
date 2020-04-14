#include <iostream>

#include "circuit.h"
#include "constants.h"

using namespace std;

extern double get_timescale(int, const string&);


Circuit::Circuit(const ModuleRegistry &module_registry): module_registry(module_registry) {
    register_01_wires();
};

Circuit::~Circuit() {
//        wires might duplicate because of assign syntax
    unordered_set<Wire*> wire_ptr_set;
    wire_ptr_set.reserve(wires.size());
    for (auto& wire_ptr: wires) {
        if (wire_ptr_set.find(wire_ptr) == wire_ptr_set.end()) {
            delete wire_ptr;
            wire_ptr_set.insert(wire_ptr);
        }
    }

    for (auto& it: cells)
        delete it.second;
}

void Circuit::summary() const {
    cout << "Summary of Circuit" << endl;
    cout << "Num cells: " << cells.size() << endl;
    cout << "Num wires: " << wires.size() << endl;
    cout << "Num schedule layers: " << cell_schedule.size() << endl;
    cout << endl;
}

Wire* Circuit::get_wire(const Wirekey& wirekey) const {
    const auto& it = wirekey_to_index.find(wirekey);
    if (it == wirekey_to_index.end())
        throw runtime_error("Wire " + wirekey.first + " index " + to_string(wirekey.second) + " not found.");
    return get_wire(it->second);
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
    for (unsigned int i = 0; i < num_wires; i++) {
        string wire_name;
        unsigned int bucket_index;
        int wire_index;
        fin >> bucket_index >> wire_name >> wire_index;
        wirekey_to_index.emplace(make_pair(wire_name, wire_index), bucket_index),
        wires.emplace_back(new Wire());
    }
}

void Circuit::read_assigns(ifstream& fin) {
    int num_assigns;
    fin >> num_assigns;
    for (int i = 0; i < num_assigns; i++) {
        unsigned int lhs_wire_index, rhs_wire_index;
        fin >> lhs_wire_index >> rhs_wire_index;
        auto lhs_wire_ptr = get_wire(lhs_wire_index), rhs_wire_ptr = get_wire(rhs_wire_index);
        delete lhs_wire_ptr;
        set_wire(lhs_wire_index, rhs_wire_ptr);  // both wirekeys now share the same Wire instance
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
        for (auto& arg: args) {
            char c;
            unsigned int wire_index;
            fin >> arg.name >> c >> wire_index;
            arg.wire = get_wire(wire_index);
        }

        unsigned int num_alloc_wires;
        fin >> num_alloc_wires;
        vector<Wire*> alloc_wires{num_alloc_wires};
        for (auto& wire_ptr: alloc_wires) {
            unsigned int wire_index;
            fin >> wire_index;
            wire_ptr = get_wire(wire_index);
        }

        unsigned int num_free_wires;
        fin >> num_free_wires;
        vector<Wire*> free_wires{num_free_wires};
        for (auto& wire_ptr: free_wires) {
            unsigned int wire_index;
            fin >> wire_index;
            wire_ptr = get_wire(wire_index);
        }

        cells.emplace(cell_name, create_cell(cell_type, args, alloc_wires, free_wires));
    }
}

Cell* Circuit::create_cell(
    const string& cell_type,
    const vector<PinSpec>& pin_specs,
    const vector<Wire*>& alloc_wires, const vector<Wire*>& free_wires
) {
    const ModuleSpec* module_spec = module_registry.get_module_spec(cell_type);
    const vector<SubmoduleSpec>* submodule_specs = module_registry.get_submodule_specs(cell_type);

    const StdCellDeclare* declare = module_registry.get_module_declare(cell_type);
    Wire *supply1_wire = get_wire(SUPPLY1_WIREKEY), *supply0_wire = get_wire(SUPPLY0_WIREKEY);
    return new Cell(
        module_spec,
        submodule_specs,
        declare,
        pin_specs, supply1_wire, supply0_wire,
        alloc_wires, free_wires
    );
}

void Circuit::read_schedules(ifstream& fin) {
    int num_schedule_layers;
    fin >> num_schedule_layers;
    cell_schedule.reserve(num_schedule_layers);
    for (int i = 0; i < num_schedule_layers; i++) {
        unsigned int num_cell;
        fin >> num_cell;
        vector<Cell*> cell_ids;
        cell_ids.reserve(num_cell);
        for (int j = 0; j < num_cell; j++) {
            string cell_id;
            fin >> cell_id;
            cell_ids.emplace_back(get_cell(cell_id));
        }
        cell_schedule.emplace_back(cell_ids);
    }
}

void Circuit::register_01_wires() {
    wirekey_to_index.emplace(make_pair("1'b0", 0), 0);
    wirekey_to_index.emplace(make_pair("1'b1", 0), 1);
    wires.push_back(new ConstantWire('0'));
    wires.push_back(new ConstantWire('1'));
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

Wire* Circuit::get_wire(const unsigned int idx) const {
    if (idx >= wires.size())
        throw runtime_error("Wire index " + to_string(idx) + " out of range");
    return wires[idx];
}

void Circuit::set_wire(unsigned int idx, Wire* wire) {
    if (idx >= wires.size())
        throw runtime_error("Wire index " + to_string(idx) + " out of range");
    wires[idx] = wire;
}
