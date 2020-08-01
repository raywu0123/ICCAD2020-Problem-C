from typing import Tuple
import warnings
from itertools import product

from graph_preprocessing.graph import Graph
from graph_preprocessing.constants import SINGLE_BIT_INDEX, BIT_INDEX_TYPE
from graph_preprocessing.utils import extract_bitwidth, is_single_bit

from .file_parsers.gv_parser import GVInfo


class Circuit:

    def __init__(self, gv_info: GVInfo, std_cell_info):
        self.std_cell_info = std_cell_info

        self.wire_keys = set()
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

    def summary(self):
        print('Circuit summary:')
        print(f"Num buses: {len(self.buses)}")
        print(f"Num wires: {len(self.wire_inputs)}")
        print(f"Num cells: {len(self.cells)}")

        print(f"Num schedule layer: {len(self.schedule_layers)}")
        all_cells_in_schedule_layers = {cell for layer in self.schedule_layers for cell in layer}
        print(f"Num cells in schedule layers: {len(all_cells_in_schedule_layers)}")

    def register_wires(self, gv_info: GVInfo):
        buses = {}
        identifiers = {}
        for wire_type, wire_info in gv_info.wire:
            if wire_info.bitwidth is not None:
                buses[wire_info.id] = wire_info.bitwidth
                identifiers[wire_info.id] = wire_info.bitwidth
                for index in range(min(wire_info.bitwidth), max(wire_info.bitwidth) + 1):
                    wire_key = self.make_wire_key(wire_info.id, index)
                    self.register_wire(wire_key)
            else:
                identifiers[wire_info.id] = (0, 0)
                wire_key = self.make_wire_key(wire_info.id, SINGLE_BIT_INDEX)
                self.register_wire(wire_key)
        return buses, identifiers

    @staticmethod
    def make_wire_key(wire_name: str, bit_index: BIT_INDEX_TYPE) -> Tuple[str, BIT_INDEX_TYPE]:
        return wire_name, bit_index

    def register_wire(self, wire_key: Tuple[str, BIT_INDEX_TYPE]):
        if wire_key in self.wire_inputs or wire_key in self.wire_outputs:
            warnings.warn(f"Wire {wire_key} duplicates in input/output/wire.")
            return
        self.wire_keys.add(wire_key)
        self.wire_inputs[wire_key] = []
        self.wire_outputs[wire_key] = []

    def register_assigns(self, gv_info: GVInfo):
        assigns = []
        for assign in gv_info.assign:
            if assign[0] == "1'b0" or assign[0] == "1'b1":
                assign = (assign[1], assign[0])

            assigns.append(assign)
            for bucket in [self.wire_inputs, self.wire_outputs]:
                bucket[assign[1]].extend(bucket[assign[0]])
                del bucket[assign[0]]

        return assigns

    def register_cells(self, gv_info: GVInfo) -> Tuple[dict, dict]:
        cells = {}
        for cell_info in gv_info.cell:
            cell_spec = self.get_cell_spec(cell_info.type)

            params = []
            for arg in cell_info.args:
                pin_type = self.get_pin_type(cell_spec, arg.pin_name)
                params.append((arg.pin_name, pin_type, arg.wire_info))
                if pin_type == "input":
                    self.wire_outputs[arg.wire_info].append(cell_info.name)
                else:
                    self.wire_inputs[arg.wire_info].append(cell_info.name)
            cells[cell_info.name] = {'parameters': params, 'type': cell_info.type}
        return cells, {cell_info.name: cell_info for cell_info in gv_info.cell}

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

