#include <iostream>
#include <simulation_result.h>

#include "circuit.h"
#include "constants.h"

using namespace std;

extern double get_timescale(int, const string&);


Circuit::Circuit(const ModuleRegistry &module_registry): module_registry(module_registry) {}

void Circuit::register_01_wires(const string& output_flag) {
    wirekey_to_index.emplace(make_pair("1'b0", 0), 0);
    wirekey_to_index.emplace(make_pair("1'b1", 0), 1);
    wires.push_back(new ConstantWire('0', output_flag));
    wires.push_back(new ConstantWire('1', output_flag));
}

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

void Circuit::read_file(ifstream &fin, double input_timescale, BusManager& bus_manager, const string& output_flag) {
    fin >> design_name;
    bus_manager.read(fin);
    register_01_wires(output_flag);
    read_wires(fin, bus_manager, output_flag);
    read_assigns(fin);
    read_cells(fin);
    read_schedules(fin);
    read_sdf(fin, input_timescale);
}

void Circuit::read_wires(ifstream& fin, BusManager&, const string& output_flag) {
    unsigned int num_wires;
    fin >> num_wires;
    wires.resize(num_wires);
    wire_buses.resize(num_wires);
    for (unsigned int i = 0; i < num_wires; i++) {
        string wire_name;
        unsigned int bucket_index;
        int wire_index, bus_index;
        fin >> bucket_index >> wire_name >> wire_index >> bus_index;
        auto wirekey = make_pair(wire_name, wire_index);
        wirekey_to_index.emplace(wirekey, bucket_index),
        wires[bucket_index] = new Wire(WireInfo{wirekey, bus_index}, output_flag);
    }
}

void Circuit::read_assigns(ifstream& fin) {
    int num_assigns;
    fin >> num_assigns;
    for (int i = 0; i < num_assigns; i++) {
        unsigned int lhs_wire_index, rhs_wire_index;
        fin >> lhs_wire_index >> rhs_wire_index;
        auto lhs_wire_ptr = get_wire(lhs_wire_index), rhs_wire_ptr = get_wire(rhs_wire_index);
        rhs_wire_ptr->assign(*lhs_wire_ptr);
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

        vector<PinSpec> args;
        args.reserve(num_args);
        for (int j = 0; j < num_args; j++) {
            char c;
            unsigned int wire_index;
            PinSpec pin_spec;
            fin >> pin_spec.name >> c >> wire_index;
            pin_spec.wire = get_wire(wire_index);
            args.push_back(pin_spec);
        }

        unsigned int num_alloc_wires;
        fin >> num_alloc_wires;
        vector<Wire*> alloc_wires;
        alloc_wires.reserve(num_alloc_wires);
        for (int j = 0; j < num_alloc_wires; j++) {
            unsigned int wire_index;
            fin >> wire_index;
            auto wire_ptr = get_wire(wire_index);
            alloc_wires.push_back(wire_ptr);
        }

        unsigned int num_free_wires;
        fin >> num_free_wires;
        vector<Wire*> free_wires;
        free_wires.reserve(num_free_wires);
        for (int j = 0; j < num_free_wires; j++) {
            unsigned int wire_index;
            fin >> wire_index;
            auto wire_ptr = get_wire(wire_index);
            free_wires.push_back(wire_ptr);
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

void BusManager::read(ifstream& fin) {
    int num_buses;
    fin >> num_buses;
    buses.resize(num_buses);
    for (int i = 0; i < num_buses; i++ ) {
        unsigned int bus_index;
        fin >> bus_index;
        fin >> buses[bus_index].name >> buses[bus_index].bitwidth.first >> buses[bus_index].bitwidth.second;
    }
}

void BusManager::add_transition(unsigned int wire_index, const Transition &transition) {

}
