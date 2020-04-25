from copy import deepcopy
from itertools import chain

from graph_preprocessing.circuit_model import Circuit
from graph_preprocessing.constants import SINGLE_BIT_INDEX


class Module:

    def __init__(self, declares):
        self.declares = declares
        self.arg_to_index_map = self.build_arg_to_index_map(declares)

    @staticmethod
    def build_arg_to_index_map(declares):
        m = {}
        for bucket in declares.values():
            for arg in bucket:
                m[arg] = len(m)
        return m

    def to_index(self, arg: str):
        return self.arg_to_index_map[arg]


class IntermediateFileWriter:

    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.module_lib = {}

    def __enter__(self):
        self.file = open(self.path, 'w')
        return self

    def print(self, *args):
        print(*args, file=self.file)

    def write_vlib_common(self, name):
        self.print(f'{name}')
        module = self.module_lib[name]
        declare_types = deepcopy(list(module.declares.keys()))
        declare_types.remove('gates')
        for declare_type in declare_types:
            self.print(
                f'{declare_type} '
                f'{len(module.declares[declare_type])} '
                f'{" ".join([str(module.to_index(arg)) for arg in module.declares[declare_type]])}'
            )

    def write_vlib_module(self, name: str, m: dict):
        self.write_vlib_common(name)
        self.print(len(m.declares['gates']))
        module = self.module_lib[name]
        for gate in chain(m.declares['gates']):
            self.print(f'gate {gate[0]} {gate[1]} {len(gate[2])} '
                       f'{" ".join([str(module.to_index(arg)) for arg in gate[2]])}')

    def write_vlib_primitive(self, name: str, m):
        self.write_vlib_common(name)
        table = [
            "".join([''.join(group) for group in row]) for row in m.table
        ]
        self.print(f'{len(table)} {" ".join(table)}')

    def write_vlib(self, std_cell_info):
        self.print(f'{len(std_cell_info.primitives)} {len(std_cell_info.modules)}')
        for name, module in chain(std_cell_info.primitives.items(), std_cell_info.modules.items()):
            assert name not in self.module_lib
            self.module_lib[name] = Module(module.declares)

        for name, m in std_cell_info.primitives.items():
            self.write_vlib_primitive(name, m)
        for name, m in std_cell_info.modules.items():
            self.write_vlib_module(name, m)

    def write_circuit(self, circuit: Circuit, design_name: str):
        self.print(design_name)
        identifier_to_index = {
            identifier: idx for idx, identifier in enumerate(circuit.identifiers)
        }
        self.print(len(circuit.identifiers))
        for identifier, bitwidth in circuit.identifiers.items():
            self.print(f'{identifier_to_index[identifier]} {identifier} {bitwidth[0]} {bitwidth[1]}')

        all_wirekeys = list(chain(*circuit.io_buckets.values()))  # not including constant wires
        wirekey_to_index = {
            ("1'b0", SINGLE_BIT_INDEX): 0,
            ("1'b1", SINGLE_BIT_INDEX): 1,
            **{
                wirekey: idx + 2 for idx, wirekey in enumerate(all_wirekeys)
            }
        }
        self.print(len(all_wirekeys))
        for wirekey in all_wirekeys:
            bus_index = identifier_to_index[wirekey[0]]
            self.print(f'{wirekey_to_index[wirekey]} {wirekey[0]} {wirekey[1]} {bus_index}')

        self.print(len(circuit.assigns))
        for assign in circuit.assigns:
            self.print(f"{wirekey_to_index[assign[0]]} {wirekey_to_index[assign[1]]}")

        self.print(len(circuit.cells))  # filtered cells, stripped to only combinational
        for cell_id, cell in circuit.cells.items():
            cell_type = cell["type"]
            module = self.module_lib[cell_type]
            self.print(f'{cell_type} {cell_id} {len(cell["parameters"])}')
            for pin_name, pin_type, wirekey in cell["parameters"]:
                self.print(f"{module.to_index(pin_name)} {pin_type[0]} {wirekey_to_index[wirekey]}")
            wirekey_schedule = circuit.mem_schedule[cell_id]
            for schedule in [wirekey_schedule["alloc"], wirekey_schedule["free"]]:
                self.print(f'{len(schedule)}')
                for wirekey in schedule:
                    self.print(f'{wirekey_to_index[wirekey]}')

        self.print(len(circuit.schedule_layers))
        for cell_ids in circuit.schedule_layers:
            self.print(len(cell_ids), ' '.join(cell_ids))

    def write_sdf(self, header, cells):
        self.print(f'timescale {header["TIMESCALE"]}')
        quotes = "'" + '"'
        cells = [c for c in cells if 'name' in c]  # the only case is the first cell block for the whole module
        self.print(len(cells))
        for cell in cells:
            cell_type = cell["type"].strip(quotes)
            module = self.module_lib[cell_type]
            self.print(f'{cell_type} {cell["name"]} {len(cell["delay"])}')
            for path in cell['delay']:
                assert len(path) == 3 or len(path) == 4  # (in out delay) or (posedge in out delay)
                if len(path) == 4:
                    assert path[0] == 'posedge' or path[0] == 'negedge'
                    path[0] = '+' if path[0] == 'posedge' else '-'
                else:
                    path.insert(0, 'x')

                assert len(path[-1]) == 3 or (len(path[-1])) == 4
                if len(path[-1]) == 3:
                    path[-1] = (path[-1][0], path[-1][0])
                else:
                    path[-1] = (path[-1][0], path[-1][3])
                self.print(
                    f'{path[0]} '
                    f'{str(module.to_index(path[1]))} '
                    f'{str(module.to_index(path[2]))} '
                    f'{path[-1][0]} {path[-1][1]}'
                )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
