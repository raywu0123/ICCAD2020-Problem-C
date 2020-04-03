from abc import ABC
from typing import List, Tuple, Dict
from itertools import chain

from .graph import Graph


SINGLE_BITWIDTH = (0, 0)
UNDEF_USAGE_BITWIDTH = (-1, -1)

BITWIDTH_TYPE = Tuple[int, int]
CELL_ID_TYPE, PIN_ID_TYPE = str, str
ASSIGNMENT_MAP_TYPE = Dict[
    PIN_ID_TYPE,
    List[Tuple[CELL_ID_TYPE, PIN_ID_TYPE, BITWIDTH_TYPE]],
]


class Node(ABC):

    def __init__(self, idd):
        self.idd = idd
        self.inputs, self.outputs = {}, {}


class SlicedWire:

    # Basically Edges

    def __init__(self, original_wire, bitwidth: BITWIDTH_TYPE, pin_type: str):
        self.original_wire = original_wire
        self.bitwidth = bitwidth
        self.inputs = set()
        self.outputs = set()

        if pin_type == 'input':
            self.inputs.add(original_wire)
        else:
            self.outputs.add(original_wire)

    def add_input(self, n: Node):
        self.inputs.add(n)

    def add_output(self, n: Node):
        self.outputs.add(n)

    def get_nodes(self, node_type: str):
        return self.inputs if node_type == 'input' else self.outputs


class WireNode(Node):

    OUTPUT_WIRES = None

    def __init__(self, idd, bitwidth=SINGLE_BITWIDTH):
        super().__init__(idd)
        self.bitwidth = bitwidth

    def add_input_slice(self, n: SlicedWire):
        self.inputs[n.bitwidth] = n

    def add_output_slice(self, n: SlicedWire):
        self.outputs[n.bitwidth] = n

    def slice(self, bitwidth: BITWIDTH_TYPE, pin_type: str):
        if bitwidth == UNDEF_USAGE_BITWIDTH:
            bitwidth = self.bitwidth
        bucket = self.inputs if pin_type == 'output' else self.outputs
        if bitwidth in bucket:
            return bucket[bitwidth]

        sliced_wire = SlicedWire(self, bitwidth, pin_type)
        bucket[bitwidth] = sliced_wire
        return sliced_wire

    def summary(self):
        print('id: ', self.idd)
        print('bitwidth: ', self.bitwidth)
        print('input slices: ', list(self.inputs.keys()))
        print('output slices: ', list(self.outputs.keys()))

    def get_io_pairs(self) -> List[Tuple[Node, Node]]:
        io_pairs = []
        for input_bitwidth, input_slice in self.inputs.items():
            for output_bitwidth, output_slice in self.outputs.items():
                if self.bitwidth_intersect(input_bitwidth, output_bitwidth):
                    io_pairs.extend([
                        (input_node, output_node)
                        for input_node in input_slice.inputs
                        for output_node in output_slice.outputs
                    ])
        if len(self.inputs) == 0:
            # being a floating wire or input wire
            for output_slice in self.outputs.values():
                for output_node in output_slice.outputs:
                    io_pairs.append((self, output_node))
        if len(self.outputs) == 0:
            # being a output wire
            for input_slice in self.inputs.values():
                for input_node in input_slice.inputs:
                    io_pairs.append((input_node, self))
        return io_pairs

    @staticmethod
    def bitwidth_intersect(b1: BITWIDTH_TYPE, b2: BITWIDTH_TYPE):
        return (min(b1) <= min(b2) <= max(b1)) or (min(b2) <= max(b1) <= max(b2))


class ConstantWireNode(WireNode):

    def __init__(self, idd, value):
        super().__init__(idd)
        self.value = value


class CellNode(Node):

    def __init__(self, idd, cell_type):
        super().__init__(idd)
        self.cell_type = cell_type

    def add_input(self, pin_name, wire: SlicedWire):
        if pin_name not in self.inputs:
            self.inputs[pin_name] = []
        self.inputs[pin_name].append(wire)

    def add_output(self, pin_name, wire: SlicedWire):
        if pin_name not in self.inputs:
            self.outputs[pin_name] = []
        self.outputs[pin_name].append(wire)


class Circuit:

    def __init__(self, gv_info, std_cell_info):
        self.std_cell_info = std_cell_info
        self.cells = {}
        self.io = {
            'input': {},
            'output': {},
            'wire': {},
        }
        self.register_constant_nodes()
        self.register_io_nodes(gv_info)

        self.register_assign_nodes(gv_info)
        self.register_cells(gv_info)

        self.graph = self.build_graph()
        self.initial_nodes = self.graph.get_initial_nodes()
        self.schedule_layers = self.graph.get_schedule_layers()
        self.mem_free_schedule = self.get_mem_free_schedule()

    def summary(self):
        print('Circuit summary:')
        print(f'Num inputs: {len(self.io["input"])}')
        floating_inputs = [i for i in self.io["input"].values() if len(i.outputs) == 0]
        print(f'\t{len(floating_inputs)} floating')

        print(f'Num outputs: {len(self.io["output"])}')
        floating_outputs = [i for i in self.io["output"].values() if len(i.inputs) == 0]
        print(f'\t{len(floating_outputs)} floating')

        for name, bucket in [('wire', self.io['wire'])]:
            print(f'Num {name}s: {len(bucket)}')
            input_floating_wires = {w for w in bucket.values() if len(w.inputs) == 0}
            print(f'\tNum input-floating {name}s: {len(input_floating_wires)}')
            output_floating_wires = {w for w in bucket.values() if len(w.outputs) == 0}
            print(f'\tNum output-floating {name}s: {len(output_floating_wires)}')
            both_end_floating_wires = set.intersection(input_floating_wires, output_floating_wires)
            print(f'\tNum both-ends-floating {name}s: {len(both_end_floating_wires)}')

        print(f'Num layers in schedule: {len(self.schedule_layers)}')
        print(f'Expected num nodes in schedule: '
              f'{len(self.io["input"]) + len(self.io["output"]) + len(self.cells) + len(input_floating_wires.union(output_floating_wires))}')
        all_nodes_in_schedule = set.union(*self.schedule_layers)
        print(f'Total nodes in schedule: {len(all_nodes_in_schedule)}')
        print(f'Expected num initial nodes in schedule: '
              f'{len(self.io["input"]) + len(input_floating_wires) + len(floating_outputs)}')
        print(f'Num initial nodes in schedule: {len(self.initial_nodes)}')

        input_nodes = set(self.io["input"].values())
        output_nodes = set(self.io["output"].values())
        print(f"\t{len(set.intersection(output_nodes, self.initial_nodes))} output nodes in initial nodes")
        for n in self.initial_nodes.difference(output_nodes).difference(input_nodes).difference(input_floating_wires):
            assert len(n.inputs) == 0

        for node_type, nodes in [
            ('input', self.io['input'].values()),
            ('output', self.io['output'].values()),
            ('input-floating wire', input_floating_wires),
            ('output-floating wire', output_floating_wires),
            ('cell', self.cells.values())
        ]:
            node_set = set(nodes)
            print(f'{len(node_set.difference(all_nodes_in_schedule))} {node_type}s not in schedule')

        all_wire_nodes_in_mem_free_schedule = set.union(*self.mem_free_schedule)
        print(f'Num wire nodes in mem free schedule: {len(all_wire_nodes_in_mem_free_schedule)}')
        assert len(self.mem_free_schedule) == len(self.schedule_layers)

    def register_constant_nodes(self):
        self.io['input']["1'b1"] = ConstantWireNode("1'b1", 1)
        self.io['input']["1'b0"] = ConstantWireNode("1'b0", 0)

    def register_io_nodes(self, gv_info):
        for io in gv_info.io:
            if 'bitwidth' in io:
                self.io[io.type][io.id] = WireNode(
                    io.id,
                    (int(io.bitwidth[0]), int(io.bitwidth[1])),
                )
            else:
                for idd in io.ids:
                    self.io[io.type][idd] = WireNode(idd)

    def register_assign_nodes(self, gv_info):
        for assign in gv_info.assign:
            wire_name, bitwidth = self.extract_bitwidth(assign[0])
            wire_node = self.get_sliced_wire(arg_name=wire_name, bitwidth=bitwidth, pin_type='input')
            value = assign[1]
            value_node = self.get_sliced_wire(arg_name=value, bitwidth=SINGLE_BITWIDTH, pin_type='output')

            value_node.outputs.add(wire_node)
            wire_node.inputs.add(value_node)

    def register_cells(self, gv_info):
        for cell in gv_info.cells:
            cell_spec = self.get_cell_spec(cell.cell_type)
            if cell_spec is None:
                continue

            cell_node = CellNode(cell.id, cell.cell_type)
            for p in cell.parameters:
                s = p.split(' ')
                pin_name = s[0]
                pin_type = self.get_pin_type(cell_spec, pin_name)
                args = [self.extract_bitwidth(arg) for arg in s[1:]]
                self.connect_pin(cell_node, pin_name, pin_type, args)

            assert cell.id not in self.cells
            self.cells[cell.id] = cell_node

    def get_cell_spec(self, cell_type: str):
        if cell_type in self.std_cell_info.primitives:
            return self.std_cell_info.primitive[cell_type]
        elif cell_type in self.std_cell_info.modules:
            return self.std_cell_info.modules[cell_type]
        else:
            return None

    def connect_pin(
            self,
            cell_node: CellNode,
            pin_name: str,
            pin_type: str,
            args: List[Tuple[str, Tuple[int, int]]],
    ):
        for arg_name, bitwidth in args:
            sliced_wire = self.get_sliced_wire(arg_name, bitwidth, pin_type)
            if pin_type == 'input':
                cell_node.add_input(pin_name, sliced_wire)
                sliced_wire.add_output(cell_node)
            else:
                cell_node.add_output(pin_name, sliced_wire)
                sliced_wire.add_input(cell_node)

    def get_sliced_wire(self, arg_name: str, bitwidth: Tuple[int, int], pin_type: str) -> SlicedWire:
        for wires in self.io.values():
            if arg_name in wires:
                return wires[arg_name].slice(bitwidth, pin_type)
        raise ValueError(f'Wire {arg_name} {bitwidth} not found.')

    @staticmethod
    def get_pin_type(cell_spec, pin_name):
        for p in cell_spec.declares['input']:
            if p == pin_name:
                return 'input'
        for p in cell_spec.declares['output']:
            if p == pin_name:
                return 'output'
        raise ValueError(f'pin {pin_name} not found.')

    @staticmethod
    def extract_bitwidth(s: str) -> (str, Tuple[int, int]):
        split = s.split('[')
        if len(split) == 1:
            return s, UNDEF_USAGE_BITWIDTH
        elif len(split) == 2:
            b = int(split[1].strip(']'))
            return split[0], (b, b)
        elif len(split) == 3:
            return split[0], (split[1], split[2].strip("]"))
        else:
            raise ValueError(f"Invalid argument: {s}")

    @staticmethod
    def add_edge(graph: Graph, wire_node: WireNode):
        for from_node, to_node in wire_node.get_io_pairs():
            try:
                graph.add_edge(from_node, to_node)
            except KeyError as err:
                print(from_node.idd, to_node.idd)
                raise err

    def build_graph(self):
        graph = Graph()
        for node in chain(self.io['input'].values(), self.io['output'].values(), self.cells.values()):
            graph.add_node(node)

        for wire_node in self.io['wire'].values():
            if len(wire_node.inputs) == 0 or len(wire_node.outputs) == 0:
                graph.add_node(wire_node)
            self.add_edge(graph, wire_node)

        for wire_node in chain(self.io['input'].values(), self.io['output'].values()):
            self.add_edge(graph, wire_node)

        return graph

    def get_mem_free_schedule(self):
        mem_free_schedule = []
        freed_wire_nodes = set()
        for layer in self.schedule_layers[::-1]:
            freed_wire_nodes_this_layer = set()

            for node in layer:
                if node in freed_wire_nodes:
                    continue

                if isinstance(node, WireNode):
                    freed_wire_nodes_this_layer.add(node)
                elif isinstance(node, CellNode):
                    for slice_wire in chain(*node.outputs.values()):
                        assert isinstance(slice_wire, SlicedWire)
                        assert isinstance(slice_wire.original_wire, WireNode)
                        if slice_wire.original_wire not in freed_wire_nodes:
                            freed_wire_nodes_this_layer.add(slice_wire.original_wire)
                else:
                    raise ValueError(f"Unknown node class: {node.__class__.__name__}")

            mem_free_schedule.append(freed_wire_nodes_this_layer)
            freed_wire_nodes.update(freed_wire_nodes_this_layer)
        return mem_free_schedule[::-1]
