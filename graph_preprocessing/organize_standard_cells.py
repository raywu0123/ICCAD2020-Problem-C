from itertools import chain, product
from copy import deepcopy
from typing import Dict, List, Tuple

from graph_preprocessing.graph import Graph


class StandardCellModule:

    @staticmethod
    def and_logic(*args) -> str:
        has_xz = False
        for arg in args:
            if arg == '0':
                return '0'
            if arg == 'x' or arg == 'z':
                has_xz = True

        if has_xz:
            return 'x'
        return '1'

    @staticmethod
    def or_logic(*args) -> str:
        has_xz = False
        for arg in args:
            if arg == '1':
                return '1'
            if arg == 'x' or arg == 'z':
                has_xz = True
        if has_xz:
            return 'x'
        return '0'

    @staticmethod
    def xor_logic(*args) -> str:
        ret = '0'
        for arg in args:
            if arg == 'x' or arg == 'z':
                return 'x'
            if ret == arg:
                ret = '0'
            else:
                ret = '1'
        return ret

    @staticmethod
    def nand_logic(*args) -> str:
        has_xz = False
        for arg in args:
            if arg == '0':
                return '1'
            if arg == 'x' or arg == 'z':
                has_xz = True
        if has_xz:
            return 'x'
        return '0'

    @staticmethod
    def nor_logic(*args) -> str:
        has_xz = False
        for arg in args:
            if arg == '1':
                return '0'
            if arg == 'x' or arg == 'z':
                has_xz = True
        if has_xz:
            return 'x'
        return '1'

    @staticmethod
    def xnor_logic(*args) -> str:
        ret = '0'
        for arg in args:
            if arg == 'x' or arg == 'z':
                return 'x'
            if ret == arg:
                ret = '0'
            else:
                ret = '1'

        if ret == '0':
            return '1'
        elif ret == '1':
            return '0'
        return ret

    @staticmethod
    def not_logic(*args) -> str:
        assert len(args) == 1
        if args[0] == '0':
            return '1'
        elif args[0] == '1':
            return '0'
        return 'x'

    @staticmethod
    def buf_logic(*args) -> str:
        assert len(args) == 1
        if args[0] == '0' or args[0] == '1':
            return args[0]
        return 'x'

    @staticmethod
    def primitive_logic(table, *args) -> str:
        ret = 'x'
        for row in table:
            all_match = True
            for table_value, input_value in zip(row[0], args):
                if input_value == 'z':
                    value = 'x'
                else:
                    value = input_value

                all_match &= table_value == '?' or table_value == value
            if all_match:
                return row[1][0]
        return ret

    def __init__(self, name: str, declares, primitives):
        self.name = name
        self.arg_declares = deepcopy(declares)
        self.gates = [self.clean_gate(gate_tuple) for gate_tuple in self.arg_declares['gates']]
        del self.arg_declares['gates']
        self.arg_to_index_map = self.build_arg_to_index_map(self.arg_declares)
        self.primitives = primitives

        self.LOGIC_FNS = {
            'and': self.and_logic,
            'or': self.or_logic,
            'xor': self.xor_logic,
            'nand': self.nand_logic,
            'nor': self.nor_logic,
            'xnor': self.xnor_logic,
            'not': self.not_logic,
            'buf': self.buf_logic,
        }
        self.table = self.get_table()

    @staticmethod
    def clean_gate(gate_tuple):
        # e.g. ('and', ['z', 'a1', 'a2'])
        return gate_tuple[0], list(gate_tuple[2])

    @staticmethod
    def build_arg_to_index_map(declares):
        m = {}
        for bucket in declares.values():
            for arg in bucket:
                m[arg] = len(m)
        return m

    def to_index(self, arg: str):
        return self.arg_to_index_map[arg]

    def get_table(self):
        table = []
        for input_state in product("01xz", repeat=len(self.arg_declares['input'])):
            output_state = self.sim(input_state)
            table.append(output_state)
        return table

    def sim(self, input_state: Tuple[str]):
        state = list(input_state) + ['x'] * (len(self.arg_to_index_map) - len(input_state))
        for w in self.arg_declares['supply1']:
            state[self.arg_to_index_map[w]] = '1'
        for w in self.arg_declares['supply0']:
            state[self.arg_to_index_map[w]] = '0'

        for gate in self.gates:
            output_arg, input_args = gate[1][0], gate[1][1:]
            gate_input = [state[self.arg_to_index_map[arg]] for arg in input_args]
            gate_name = gate[0]
            if gate_name in self.LOGIC_FNS:
                gate_output = self.LOGIC_FNS[gate[0]](*gate_input)
            else:
                gate_output = self.primitive_logic(self.primitives[gate_name].table, *gate_input)
            state[self.arg_to_index_map[output_arg]] = gate_output

        num_outputs = len(self.arg_declares['output'])
        return ''.join(state[len(input_state): len(input_state) + num_outputs])

    def __str__(self):
        decl = f"{self.name} " \
               f"{len(self.arg_declares['input'])} " \
               f"{len(self.arg_declares['output'])}\n"
        table_str = '\n'.join([f'{table_row}' for table_row in self.table])
        return decl + table_str


class StandardCellOrganizer:

    def __init__(self, primitives, modules, used_cell_types):
        self.primitives = primitives
        if used_cell_types is not None:
            self.modules = {
                module_name: module_spec
                for module_name, module_spec in modules.items()
                if module_name in used_cell_types
            }
        else:
            self.modules = modules

    def organize(self) -> Dict[str, StandardCellModule]:
        return {
            module_name: self.organize_module(module_name, module_spec.declares)
            for module_name, module_spec in self.modules.items()
        }

    def organize_module(self, module_name: str, module_declares: dict) -> StandardCellModule:
        graph = Graph()
        for gate_idx, gate in enumerate(module_declares['gates']):
            assert len(gate) == 2
            # (gate_type, args) -> (gate_type, gate_id, args)
            module_declares['gates'][gate_idx] = (gate[0], gate_idx, gate[1])

        all_gates = lambda: chain(module_declares['gates'], module_declares['submodules'])
        all_wires = lambda: chain(module_declares['wire'], module_declares['supply1'], module_declares['supply0'])

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

        return StandardCellModule(module_name, module_declares, self.primitives)
