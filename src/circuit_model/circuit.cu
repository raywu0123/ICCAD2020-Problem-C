#include <iostream>
#include "sstream"

#include "circuit.h"
#include "constants.h"

using namespace std;

extern double get_timescale(int, const string&);

string BusManager::index_to_identifier(unsigned int index) {
//    TODO more similar to sample output format
    stringstream ss;
    ss << hex << index;
    return ss.str();
}

string BusManager::dumps_token_to_bus_map(const unordered_set<string>& referenced_wire_ids) const {
    stringstream ss;
    for (unsigned int bus_index = 0; bus_index < buses.size(); bus_index++) {
        const auto& bus = buses[bus_index];
        if (referenced_wire_ids.find(bus.name) == referenced_wire_ids.end()) continue;
        const auto& identifier = index_to_identifier_map[bus_index];
        unsigned int bits = abs(bus.bitwidth.first - bus.bitwidth.second) + 1;
        ss << "$var wire " << bits << " " << identifier << " " << bus.name;
        if (bus.bitwidth != make_pair(0, 0)) {
            ss << " [" << bus.bitwidth.first << ":" << bus.bitwidth.second << "]";
        }
        ss << " $end" << endl;
    }
    return ss.str();
}

void BusManager::write_init(const std::vector<Wire*>& wires) {
    for (const auto* wire : wires) {
        add_transition(wire->wire_infos, Transition{0, Values::X});
    }
}

void BusManager::dumpon_init(const vector<Wire*>& wires) {
    for (const auto* wire : wires) {
        for (const auto& wire_info : wire->wire_infos) {
            auto& bus = buses[wire_info.bus_index];
            if (bus.is_not_initial_state()) {
                used_buses_in_current_time.emplace(&bus, wire_info.bus_index);
            }
        }
    }
}

void BusManager::add_transition(const vector<WireInfo>& wire_infos, const Transition& transition) {
    for (const auto& wire_info: wire_infos) {
        auto& bus = buses[wire_info.bus_index];
        bus.update(transition, wire_info.wirekey.second);
        used_buses_in_current_time.emplace(&bus, wire_info.bus_index);
    }
}

std::string BusManager::dumps_result(const Timestamp& t) {
    if (used_buses_in_current_time.empty()) return "";
    stringstream ss;
    ss << "#" << t << endl;
    for (const auto& p : used_buses_in_current_time) {
        const auto* bus = p.first;
        const auto& bus_index = p.second;
        const auto& bus_identifier = index_to_identifier_map[bus_index];
        if (bus->bitwidth.first == bus->bitwidth.second) {
            ss << bus->state << bus_identifier << endl;
        } else {
            ss << "b" << simplify_msb(bus->state) << " " << bus_identifier << endl;
        }
    }
    used_buses_in_current_time.clear();
    return ss.str();
}

std::string BusManager::simplify_msb(const std::string& full_state) {
    if (full_state[0] == '1') return full_state;
    auto first_not_of_idx = full_state.find_first_not_of(full_state[0]);

    if (first_not_of_idx == string::npos) return full_state.substr(full_state.size() - 1);
    if (full_state[0] == '0' and full_state[first_not_of_idx] == '1') return full_state.substr(first_not_of_idx);
    return full_state.substr(first_not_of_idx - 1);
}

Circuit::Circuit(const ModuleRegistry &module_registry): module_registry(module_registry) {}

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

Wire* Circuit::get_wire(const Wirekey& wirekey) {
    // used by vcd reader
    const auto& it = wirekey_to_index.find(wirekey);
    if (it == wirekey_to_index.end())
        throw runtime_error("Wire " + wirekey.first + " index " + to_string(wirekey.second) + " not found.");

    return get_wire(it->second);
}

Wire* Circuit::get_wire(const unsigned int idx) {
    if (idx >= wires.size())
        throw runtime_error("Wire index " + to_string(idx) + " out of range");

    auto* wire = wires[idx];
    for (const auto& info : wire->wire_infos) {
        referenced_wire_ids.insert(info.wirekey.first);
    }
    return wire;
}

void Circuit::set_wire(unsigned int idx, Wire* wire) {
    if (idx >= wires.size())
        throw runtime_error("Wire index " + to_string(idx) + " out of range");
    wires[idx] = wire;
}

Cell* Circuit::get_cell(const string& name) const {
    const auto& it = cells.find(name);
    if (it == cells.end())
        throw runtime_error("Cell " + name + " not found");
    return it->second;
}

void Circuit::read_intermediate_file(ifstream &fin, double input_timescale, BusManager& bus_manager) {
    cout << "| STATUS: Reading intermediate file..." << endl;
    fin >> design_name;
    bus_manager.read(fin);
    register_01_wires();
    read_wires(fin);
    read_assigns(fin);
    read_cells(fin);
    read_schedules(fin);
    read_sdf(fin, input_timescale);
}

void Circuit::register_01_wires() {
    wirekey_to_index.emplace(make_pair("1'b0", 0), 0);
    wirekey_to_index.emplace(make_pair("1'b1", 0), 1);
    wires.push_back(new ConstantWire(Values::ZERO));
    wires.push_back(new ConstantWire(Values::ONE));
}

void BusManager::read(ifstream& fin) {
    int num_buses;
    fin >> num_buses;
    buses.resize(num_buses);
    index_to_identifier_map.resize(num_buses);
    for (int i = 0; i < num_buses; i++ ) {
        unsigned int bus_index;
        fin >> bus_index;
        string name;
        BitWidth bitwidth;
        fin >> name >> bitwidth.first >> bitwidth.second;
        buses[bus_index].init(name, bitwidth);
        index_to_identifier_map[bus_index] = index_to_identifier(bus_index);
    }
}

void Circuit::read_wires(ifstream& fin) {
    unsigned int num_wires;
    fin >> num_wires;
    wires.resize(2 + num_wires);  // plus two for constant wires
    for (unsigned int i = 0; i < num_wires; i++) {
        string wire_name;
        unsigned int bucket_index;
        int wire_index, bus_index;
        fin >> bucket_index >> wire_name >> wire_index >> bus_index;
        auto wirekey = make_pair(wire_name, wire_index);
        wirekey_to_index.emplace(wirekey, bucket_index),
        wires[bucket_index] = new Wire(WireInfo{wirekey, bus_index});
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

        WireMap<Wire> args{};
        for (unsigned int j = 0; j < num_args; j++) {
            unsigned int arg, wire_index;
            fin >> arg >> wire_index;
            auto* wire = get_wire(wire_index);
            args.set(arg, wire);
        }
        const ModuleSpec* module_spec = module_registry.get_module_spec(cell_type);
        const StdCellDeclare* declare = module_registry.get_module_declare(cell_type);
        cells.emplace(cell_name, new Cell(module_spec, declare, args, cell_name));
    }
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
        for (unsigned int j = 0; j < num_cell; j++) {
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

        auto& paths = get_cell(name)->sdf_paths;
        paths.resize(num_paths);
        for(int i_path = 0; i_path < num_paths; i_path++) {
            auto& path = paths[i_path];
            double sdf_rising_delay, sdf_falling_delay;
            unsigned int in, out;
            fin >> path.edge_type >> in >> out >> sdf_rising_delay >> sdf_falling_delay;
            path.in = in; path.out = out;

            // convert to VCD specified time unit
            path.rising_delay = (int)(sdf_rising_delay * sdf_timescale / input_timescale);
            path.falling_delay = (int)(sdf_falling_delay * sdf_timescale / input_timescale);
        }
    }
}

vector<Wire*> Circuit::get_referenced_wires() const {
    vector<Wire*> ret;
    for (auto* w : wires) {
        for (const auto& wire_info : w->wire_infos) {
            if (referenced_wire_ids.find(wire_info.wirekey.first) != referenced_wire_ids.end()) {
                ret.push_back(w);
                break;
            }
        }
    }
    return ret;
}

void Bus::init(const string& name_param, const BitWidth& bitwidth_param) {
    name = name_param;
    bitwidth = bitwidth_param;
    int max_bit_index = max(bitwidth.first, bitwidth.second);
    int min_bit_index = min(bitwidth.first, bitwidth.second);
    state.reserve(max_bit_index - min_bit_index + 1);
    for (int i = min_bit_index; i <= max_bit_index; i++) {
        state.push_back('x');
    }
}

void Bus::update(const Transition &transition, int index) {
//  convert bus index to array index
//  in convenience of dumping out values, array starts from MSB
    unsigned int array_index = abs(index - bitwidth.first);
    if (array_index >= state.size())
        throw runtime_error("Array index " + to_string(array_index) + " out of bounds.");
    state[array_index] = enum_to_raw(transition.value);
}

bool Bus::is_not_initial_state() const {
    for (const auto& s : state) {
        if (s != 'x') return true;
    }
    return false;
}
