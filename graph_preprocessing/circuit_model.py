from typing import Tuple, Dict
import warnings
from itertools import product

from graph_preprocessing.graph import Graph
from graph_preprocessing.constants import SINGLE_BIT_INDEX, BIT_INDEX_TYPE
from graph_preprocessing.utils import extract_bitwidth, is_single_bit


class Circuit:

    def __init__(self, gv_info, std_cell_info):
        self.std_cell_info = std_cell_info

        self.io_buckets = {
            'input': set(),
            'output': set(),
            'wire': set(),
        }
        self.wire_inputs = {
            ("1'b1", SINGLE_BIT_INDEX): [],
            ("1'b0", SINGLE_BIT_INDEX): [],
        }   # wire_id -> List[cell_id]
        self.wire_outputs = {
            ("1'b1", SINGLE_BIT_INDEX): [],
            ("1'b0", SINGLE_BIT_INDEX): [],
        }

        self.buses, self.identifiers = self.register_wires(gv_info)
        self.cells, self.cell_id_to_cell = self.register_cells(gv_info)
        self.assigns = self.register_assigns(gv_info)

        self.graph = self.build_graph()
        self.schedule_layers = [list(layer) for layer in self.graph.get_schedule_layers()]
        self.mem_schedule = self.get_mem_schedules()

    def summary(self):
        print('Circuit summary:')
        print(f"Num buses: {len(self.buses)}")
        print(f"Num wires: {len(self.wire_inputs)}")
        print(f"Num cells: {len(self.cells)}")

        print(f"Num input keys: {len(self.io_buckets['input'])}")
        print(f"Num output keys: {len(self.io_buckets['output'])}")

        print(f"Num schedule layer: {len(self.schedule_layers)}")
        all_cells_in_schedule_layers = {cell for layer in self.schedule_layers for cell in layer}
        print(f"Num cells in schedule layers: {len(all_cells_in_schedule_layers)}")

    def register_wires(self, gv_info):
        buses = {}
        identifiers = {}
        for io in gv_info.io:
            if 'bitwidth' in io:
                buses[io.id] = io.bitwidth
                identifiers[io.id] = io.bitwidth
                for index in range(min(io.bitwidth), max(io.bitwidth) + 1):
                    wire_key = self.make_wire_key(io.id, index)
                    self.register_wire(wire_key)
                    self.io_buckets[io.type].add(wire_key)
            else:
                for idd in io.ids:
                    identifiers[idd] = (0, 0)
                    wire_key = self.make_wire_key(idd, SINGLE_BIT_INDEX)
                    self.register_wire(wire_key)
                    self.io_buckets[io.type].add(wire_key)
        return buses, identifiers

    @staticmethod
    def make_wire_key(wire_name: str, bit_index: BIT_INDEX_TYPE) -> Tuple[str, BIT_INDEX_TYPE]:
        return wire_name, bit_index

    def register_wire(self, wire_key: Tuple[str, BIT_INDEX_TYPE]):
        if wire_key in self.wire_inputs or wire_key in self.wire_outputs:
            warnings.warn(f"Wire {wire_key} duplicates in input/output/wire.")
            return
        self.wire_inputs[wire_key] = []
        self.wire_outputs[wire_key] = []

    def register_assigns(self, gv_info):
        assigns = []
        for assign in gv_info.assign:
            if assign[0] == "1'b0" or assign[0] == "1'b1":
                assign = (assign[1], assign[0])
            lhs_name, lhs_bitwidth = extract_bitwidth(assign[0])
            rhs_name, rhs_bitwidth = extract_bitwidth(assign[1])
            if lhs_name in self.buses:
                assert is_single_bit(lhs_bitwidth)
                lhs_key = self.make_wire_key(lhs_name, lhs_bitwidth[0])
            else:
                lhs_key = self.make_wire_key(lhs_name, SINGLE_BIT_INDEX)
            if rhs_name in self.buses:
                assert is_single_bit(rhs_bitwidth)
                rhs_key = self.make_wire_key(rhs_name, rhs_bitwidth[0])
            else:
                rhs_key = self.make_wire_key(rhs_name, SINGLE_BIT_INDEX)

            assigns.append((lhs_key, rhs_key))
            for bucket in [self.wire_inputs, self.wire_outputs]:
                bucket[rhs_key].extend(bucket[lhs_key])
                del bucket[lhs_key]

        return assigns

    def register_cells(self, gv_info) -> Tuple[dict, dict]:
        cells = {}
        for cell in gv_info.cells:
            cell_spec = self.get_cell_spec(cell.cell_type)
            if cell_spec is None:
                continue

            params = []
            for p in cell.parameters:
                pin_name, wire_key = self.from_parameter_string(p)
                pin_type = self.get_pin_type(cell_spec, pin_name)
                params.append((pin_name, pin_type, wire_key))
                if pin_type == "input":
                    self.wire_outputs[wire_key].append(cell.id)
                else:
                    self.wire_inputs[wire_key].append(cell.id)
            cells[cell.id] = {'parameters': params, 'type': cell.cell_type}
        return cells, {cell.id: cell for cell in gv_info.cells}

    def from_parameter_string(self, p: str):
        try:
            pin_name, wire = p.split(' ')
        except ValueError as e:
            print(p)
            raise e
        wire_name, bitwidth = extract_bitwidth(wire)
        if wire_name in self.buses:
            assert is_single_bit(bitwidth)
            wire_key = self.make_wire_key(wire_name, bitwidth[0])
        else:
            wire_key = self.make_wire_key(wire_name, SINGLE_BIT_INDEX)
        return pin_name, wire_key

    def get_cell_spec(self, cell_type: str):
        if cell_type in self.std_cell_info.primitives:
            return self.std_cell_info.primitives[cell_type]
        elif cell_type in self.std_cell_info.modules:
            return self.std_cell_info.modules[cell_type]
        else:
            return None

    @staticmethod
    def get_pin_type(cell_spec, pin_name):
        for p in cell_spec.declares['input']:
            if p == pin_name:
                return 'input'
        for p in cell_spec.declares['output']:
            if p == pin_name:
                return 'output'
        raise ValueError(f'pin {pin_name} not found.')

    def build_graph(self):
        graph = Graph()
        for cell_id in self.cells:
            graph.add_node(cell_id)

        assert set(self.wire_inputs.keys()) == set(self.wire_outputs.keys())
        for k in self.wire_inputs.keys():
            for in_cell, out_cell in product(self.wire_inputs[k], self.wire_outputs[k]):
                graph.add_edge(in_cell, out_cell)
        return graph

    def get_mem_schedules(self) -> Dict[str, Dict[str, list]]:
        schedule = {cell_id: {"alloc": set(), "free": set()} for cell_id in self.cells}

        alloc_discovered_wirekeys = set()
        for layer in self.schedule_layers:
            for cell_id in layer:
                using_wirekeys = {self.from_parameter_string(p)[1] for p in self.cell_id_to_cell[cell_id].parameters}
                using_wirekeys.difference_update(alloc_discovered_wirekeys)  # unallocated wirekeys
                schedule[cell_id]["alloc"] = using_wirekeys
                alloc_discovered_wirekeys.update(using_wirekeys)

        free_discovered_wirekeys = set()
        for layer in reversed(self.schedule_layers):
            for cell_id in reversed(layer):
                using_wirekeys = {self.from_parameter_string(p)[1] for p in self.cell_id_to_cell[cell_id].parameters}
                using_wirekeys.difference_update(free_discovered_wirekeys)  # unfreed wirekeys
                schedule[cell_id]['free'] = using_wirekeys
                free_discovered_wirekeys.update(using_wirekeys)

        return schedule
