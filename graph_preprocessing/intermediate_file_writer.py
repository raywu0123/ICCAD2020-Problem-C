from copy import deepcopy
from itertools import chain

from graph_preprocessing.circuit_model import Circuit


class IntermediateFileWriter:

    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.gv_io = {'input': [], 'output': [], 'wire': []}
        self.multibit_gv_io = {'input': [], 'output': [], 'wire': []}

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

    def write_graph(self, g: Circuit):
        pass

    # def write_gv(self, gv_info):
    #     for io in gv_info.io:
    #         if 'bitwidth' in io:
    #             self.multibit_gv_io[io.type].append(f'{io.id} {io.bitwidth[0]} {io.bitwidth[1]}')
    #         else:
    #             self.gv_io[io.type].extend(io.ids)
    #
    #     self.print(gv_info.id)
    #     for io_type in self.gv_io.keys():
    #         self.print(f'{io_type} {len(self.gv_io[io_type])} {" ".join(self.gv_io[io_type])}')
    #         self.print(f'multibit_{io_type} {len(self.multibit_gv_io[io_type])} {" ".join(self.multibit_gv_io[io_type])}')
    #
    #     self.print(len(gv_info.assign))
    #     for assign in gv_info.assign:
    #         self.print(f"{self.extract_bitwidth(assign[0])} {assign[1]}")
    #
    #     self.print(len(gv_info.cells))
    #     for cell in gv_info.cells:
    #         self.print(f'{cell.cell_type} {cell.id} {len(cell.parameters)}')
    #         for p in cell.parameters:
    #             s = p.split(' ')
    #             self.print(f'{s[0]} {len(s[1:])}')
    #             args = [self.extract_bitwidth(arg) for arg in s[1:]]
    #             self.print(" ".join(args))

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
