from pprint import pprint
from itertools import chain, product

from graph_preprocessing.graph import Graph


class Gate:

    def __init__(self, idd):
        self.idd = idd


class StandardCellOrganizer:

    def __init__(self, std_cell_info):
        self.std_cell_info = std_cell_info

    def organize_primitives(self):
        for p in self.std_cell_info.primitives.values():
            del p.declares['submodules']

    def organize_modules(self):
        modules = self.std_cell_info.modules
        for module_spec in modules.values():
            self.organize_module(module_spec.declares)

    def organize_module(self, module_declares: dict):
        graph = Graph()
        for gate_idx, gate in enumerate(module_declares['gates']):
            assert len(gate) == 2
            # (gate_type, args) -> (gate_type, gate_id, args)
            module_declares['gates'][gate_idx] = (gate[0], gate_idx, gate[1])

        all_gates = lambda : chain(module_declares['gates'], module_declares['submodules'])
        all_wires = lambda : chain(module_declares['wire'], module_declares['supply1'], module_declares['supply0'])

        tuple_to_gate = {}
        for gate in all_gates():
            graph.add_node(tuple(gate[:2]))
            tuple_to_gate[tuple(gate[:2])] = gate

        wire_outputs = {
            w: [tuple(gate[:2]) for gate in all_gates() if w in gate[2][1:]]
            for w in all_wires()
        }
        wire_inputs = {
            w: [tuple(gate[:2]) for gate in all_gates() if w == gate[2][0]]
            for w in all_wires()
        }
        for w in all_wires():
            for wi, wo in product(wire_inputs[w], wire_outputs[w]):
                graph.add_edge(wi, wo)

        topological_layers = graph.get_schedule_layers()
        topological_order = list(chain(*topological_layers))

        new_gates = [tuple_to_gate[t] for t in topological_order]
        module_declares['gates'] = new_gates
        del module_declares['submodules']
