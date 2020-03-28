from copy import deepcopy


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

    def write_vlib_common(self, name, m, primitive: bool):
        declare_types = deepcopy(list(m.declares.keys()))
        declare_types.remove('gates')
        declare_types.remove('submodules')

        len_declare = declare_types if primitive else m.declares
        self.print(f'{name} {" ".join([str(len(m.declares[d])) for d in len_declare])}')
        for declare_type in declare_types:
            self.print(f'{declare_type} {" ".join(m.declares[declare_type])}')

    def write_vlib_module(self, name: str, m: dict):
        self.write_vlib_common(name, m, primitive=False)
        for gate in m.declares['gates']:
            self.print(f'gate {gate[0]} {" ".join(gate[1])}')
        for submod in m.declares['submodules']:
            submod[-1] = ' '.join(submod[-1])
            self.print(f'submodule {" ".join(submod)}')
        self.print('endmodule')

    def write_vlib_primitive(self, name: str, m):
        self.write_vlib_common(name, m, primitive=True)
        table = [
            " ".join([''.join(group) for group in row]) for row in m.table
        ]
        for row in table:
            self.print(row)
        self.print('endprimitive')

    def write_vlib(self, std_cell_info):
        for name, m in std_cell_info.primitives.items():
            self.write_vlib_primitive(name, m)
        for name, m in std_cell_info.modules.items():
            self.write_vlib_module(name, m)
        self.print('endvlib')

    def write_gv(self, gv_info):
        for io in gv_info.io:
            if 'bitwidth' in io:
                self.multibit_gv_io[io.type].append(f'{io.id} {io.bitwidth[0]} {io.bitwidth[1]}')
            else:
                self.gv_io[io.type].extend(io.ids)
        self.print(gv_info.id)
        for io_type in self.gv_io.keys():
            self.print(f'{io_type} {" ".join(self.gv_io[io_type])}')
            self.print(f'multibit_{io_type} {" ".join(self.multibit_gv_io[io_type])}')

        for cell in gv_info.cells:
            self.print(f'{cell.cell_type} {cell.id} {len(cell.parameters)}')
            for p in cell.parameters:
                self.print(p)
            self.print('endcell')
        self.print('endgv')

    def write_sdf(self, header, cells):
        self.print(f'timescale {header["TIMESCALE"]}')
        quotes = "'" + '"'
        for cell in cells:
            if 'name' not in cell:
                continue  # the only case is the first cell block for the whole module
            self.print(f'{cell["type"].strip(quotes)} {cell["name"]} {len(cell["delay"])}')
            for path in cell['delay']:
                rising_delay = path[-1][0]
                if len(path[-1]) < 4:
                    self.print(f'{" ".join(path[:-1])} {rising_delay} {rising_delay}')
                else:
                    self.print(f'{" ".join(path[:-1])} {rising_delay} {path[-1][3]}')
        self.print('endsdf')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
