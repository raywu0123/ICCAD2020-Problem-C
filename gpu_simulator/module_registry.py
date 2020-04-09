from .data_structures import (
    ANDGate,
    ORGate,
    NOTGate,
    XORGate,
    XNORGate,
    NANDGate,
    NORGate,
    BUFGate,
    Primitive,
    Module,
)


class ModuleRegistry:

    def __init__(self, std_cell_info):
        self.gates = {
            'and': ANDGate,
            'or': ORGate,
            'not': NOTGate,
            'xor': XORGate,
            'xnor': XNORGate,
            'nand': NANDGate,
            'nor': NORGate,
            'buf': BUFGate,
        }
        self.modules = {}
        for name, p in std_cell_info.primitives.items():
            self.register_primitive(name, p)
        for name, m in std_cell_info.modules.items():
            self.register_module(name, m)

    def register_primitive(self, name: str, p):
        if name in self.gates:
            raise ValueError(f"Duplicate primitives: {name}")
        self.gates[name] = Primitive(p.declares, p.table)

    def register_module(self, name: str, m):
        if name in self.modules:
            raise ValueError(f"Duplicate modules: {name}")

        processed_gates = [
            (self.gates[gate[0]], *gate[1:])
            for gate in m.declares['gates']
        ]
        self.modules[name] = Module(m.declares['input'], m.declares['output'], processed_gates)
