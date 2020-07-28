from itertools import chain
from typing import Dict

from graph_preprocessing.circuit_model import Circuit
from graph_preprocessing.constants import SINGLE_BIT_INDEX
from .organize_standard_cells import StandardCellModule


class IntermediateFileWriter:

    def __init__(self, path: str):
        self.path = path
        self.file = None
        self.module_lib = None

    def __enter__(self):
        self.file = open(self.path, 'w')
        return self

    def print(self, *args):
        print(*args, file=self.file)

    def write_vlib(self, standard_cell_library: Dict[str, StandardCellModule]):
        self.module_lib = standard_cell_library
        self.print(f'{len(standard_cell_library)}')
        for cell in standard_cell_library.values():
            self.print(str(cell))

    def write_circuit(self, circuit: Circuit, design_name: str):
        self.print(design_name)
        identifier_to_index = {
            identifier: idx for idx, identifier in enumerate(circuit.identifiers)
        }
        self.print(len(circuit.identifiers))
        for identifier, bitwidth in circuit.identifiers.items():
            self.print(f'{identifier_to_index[identifier]} {identifier} {bitwidth[0]} {bitwidth[1]}')

        all_wirekeys = list(set(chain(*circuit.io_buckets.values())))  # not including constant wires
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
            for pin_name, _, wirekey in cell["parameters"]:
                self.print(f"{module.to_index(pin_name)} {wirekey_to_index[wirekey]}")

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

                if not (len(path[-1]) == 3 or len(path[-1]) == 6):
                    raise ValueError(f"Invalid path {path[-1]}")
                if len(path[-1]) == 3:
                    path[-1] = (path[-1][1], path[-1][1])
                else:
                    path[-1] = (path[-1][1], path[-1][4])
                self.print(
                    f'{path[0]} '
                    f'{str(module.to_index(path[1]))} '
                    f'{str(module.to_index(path[2]))} '
                    f'{path[-1][0]} {path[-1][1]}'
                )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
