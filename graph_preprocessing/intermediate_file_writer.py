from copy import deepcopy
from itertools import chain

from graph_preprocessing.circuit_model import Circuit


class IntermediateFileWriter:

    def __init__(self, path: str):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = open(self.path, 'w')
        return self

    def print(self, *args):
        print(*args, file=self.file)

    def write_vlib_common(self, name, m):
        declare_types = deepcopy(list(m.declares.keys()))
        declare_types.remove('gates')

        self.print(f'{name}')
        for declare_type in declare_types:
            self.print(f'{declare_type} {len(m.declares[declare_type])} {" ".join(m.declares[declare_type])}')

    def write_vlib_module(self, name: str, m: dict):
        self.write_vlib_common(name, m)
        self.print(len(m.declares['gates']))
        for gate in chain(m.declares['gates']):
            self.print(f'gate {gate[0]} {gate[1]} {len(gate[2])} {" ".join(gate[2])}')

    def write_vlib_primitive(self, name: str, m):
        self.write_vlib_common(name, m)
        table = [
            "".join([''.join(group) for group in row]) for row in m.table
        ]
        self.print(f'{len(table)} {" ".join(table)}')

    def write_vlib(self, std_cell_info):
        self.print(f'{len(std_cell_info.primitives)} {len(std_cell_info.modules)}')
        for name, m in std_cell_info.primitives.items():
            self.write_vlib_primitive(name, m)
        for name, m in std_cell_info.modules.items():
            self.write_vlib_module(name, m)

    def write_circuit(self, circuit: Circuit, design_name: str):
        self.print(design_name)
        self.print(sum([len(bucket) for bucket in circuit.io_buckets.values()]))
        for wire_key in chain(*circuit.io_buckets.values()):
            self.print(f'{wire_key[0]} {wire_key[1]}')

        self.print(len(circuit.assigns))
        for assign in circuit.assigns:
            self.print(f"{assign[0][0]} {assign[0][1]} {assign[1][0]} {assign[1][1]}")

        self.print(len(circuit.cells))  # filtered cells, stripped to only combinational
        for cell_id, cell in circuit.cells.items():
            self.print(f'{cell["type"]} {cell_id} {len(cell["parameters"])}')
            for pin_name, pin_type, wire_key in cell["parameters"]:
                self.print(f"{pin_name} {pin_type[0]} {wire_key[0]} {wire_key[1]}")

        self.print(len(circuit.schedule_layers))
        for cell_ids, alloc_wire_keys, free_wire_keys in zip(
                circuit.schedule_layers, circuit.mem_alloc_schedule, circuit.mem_free_schedule
        ):
            self.print(len(cell_ids), ' '.join(cell_ids))
            self.print(len(alloc_wire_keys), ' '.join([f'{wire_key[0]} {wire_key[1]}' for wire_key in alloc_wire_keys]))
            self.print(len(free_wire_keys), ' '.join([f'{wire_key[0]} {wire_key[1]}' for wire_key in free_wire_keys]))
            self.print()

    def write_sdf(self, header, cells):
        self.print(f'timescale {header["TIMESCALE"]}')
        quotes = "'" + '"'
        cells = [c for c in cells if 'name' in c]  # the only case is the first cell block for the whole module
        self.print(len(cells))
        for cell in cells:
            self.print(f'{cell["type"].strip(quotes)} {cell["name"]} {len(cell["delay"])}')
            for path in cell['delay']:
                assert len(path) == 3 or len(path) == 4  # (in out delay) or (posedge in out delay)
                if len(path) == 4:
                    assert path[0] == 'posedge' or path[0] == 'negedge'
                    sign = '+' if path[0] == 'posedge' else '-'
                    path = [f'{sign}{path[1]}', path[2], path[3]]

                rising_delay = path[-1][0]
                if len(path[-1]) < 4:
                    self.print(f'{" ".join(path[:-1])} {rising_delay} {rising_delay}')
                else:
                    self.print(f'{" ".join(path[:-1])} {rising_delay} {path[-1][3]}')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
