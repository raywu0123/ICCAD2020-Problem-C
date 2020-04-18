#include <iostream>

#include "cell.h"

using namespace std;


Cell::Cell(
    const ModuleSpec* module_spec,
    const vector<SubmoduleSpec>* submodule_specs,
    const StdCellDeclare* declare,
    const vector<PinSpec> &pin_specs,
    Wire* supply1_wire, Wire* supply0_wire,
    vector<Wire*> alloc_wires_param, vector<Wire*> free_wires_param
) : module_spec(module_spec), alloc_wires(std::move(alloc_wires_param)), free_wires(std::move(free_wires_param))
{
    alloc_wires_size = alloc_wires.size();
    free_wires_size = free_wires.size();
    const auto& wire_map = build_wire_map(declare, pin_specs, supply1_wire, supply0_wire);
    create_wire_schedule(submodule_specs, wire_map);
}

Cell::~Cell() {
    for (auto& wire_ptr: cell_wires) {
        delete wire_ptr;
    }
}

unordered_map<string, Wire *> Cell::build_wire_map(
    const StdCellDeclare* declare,
    const vector<PinSpec> &pin_specs,
    Wire *supply1_wire, Wire *supply0_wire)
{
    unordered_map<string, Wire*> wire_map;
    for (const auto& pin_spec: pin_specs) wire_map[pin_spec.name] = pin_spec.wire;
    for (const auto& arg: declare->buckets[STD_CELL_SUPPLY1]) wire_map[arg] = supply1_wire;
    for (const auto& arg: declare->buckets[STD_CELL_SUPPLY0]) wire_map[arg] = supply0_wire;
    return wire_map;
}


void Cell::create_wire_schedule(
    const vector<SubmoduleSpec>* submodule_specs,
    const unordered_map<string, Wire*>& wire_map
)  {
    for(const auto& submodule_spec: *submodule_specs) {
        for (const auto& arg: submodule_spec.args) {
            const auto& it = wire_map.find(arg);
            if (it != wire_map.end()) {
                wire_schedule.push_back(it->second);
            } else {
                Wire* wire_ptr = new Wire();
                wire_schedule.push_back(wire_ptr);
                add_cell_wire(wire_ptr);
            }
        }
    }
}

void Cell::add_cell_wire(Wire *wire_ptr) {
    cell_wires.push_back(wire_ptr);
    alloc_wires.push_back(wire_ptr);
    alloc_wires_size++;
    free_wires.push_back(wire_ptr);
    free_wires_size++;
}

void Cell::prepare_resource(ResourceBuffer& resource_buffer)  {
    for (unsigned i = 0; i < alloc_wires_size; i++) {
        const auto& wire_ptr = alloc_wires[i];
        wire_ptr->alloc();
    }
    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.data_schedule_offsets.push_back(resource_buffer.data_schedule_offsets.size());
    for (const auto& wire : wire_schedule) {
        resource_buffer.data_schedule.push_back(wire->data_ptr);
        resource_buffer.capacities.push_back(wire->capacity);
    }
}

void Cell::free_resource() {
    for (unsigned i = 0; i < free_wires_size; i++) {
        const auto& wire_ptr = free_wires[i];
        wire_ptr->free();
    }
}