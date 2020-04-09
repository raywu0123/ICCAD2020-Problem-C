from sys import argv
import pickle

from graph_preprocessing.file_parsers import VlibReader, SDFParser, GVParser
from gpu_simulator.module_registry import ModuleRegistry

from graph_preprocessing.circuit_model import Circuit
from graph_preprocessing.organize_standard_cells import StandardCellOrganizer


if __name__ == '__main__':
    if len(argv) != 5:
        raise ValueError(
            "Usage: file_parsers.cu.py "
            "<netlist.gv> "
            "<netlist.SDF> "
            "<std_cells.vlib> "
            "[intermediate_representation.file] "
        )

    netlist_gv_file, netlist_sdf_file, std_cells_file, output_file = argv[1:]

    if std_cells_file != '-':
        print('Reading standard cell library... ', end='', flush=True)
        std_cell_info = VlibReader.read_file(std_cells_file)
        print('Finished.')
        standard_cell_organizer = StandardCellOrganizer(std_cell_info)
        standard_cell_organizer.organize_modules()
        standard_cell_organizer.organize_primitives()
        module_registry = ModuleRegistry(std_cell_info)
    else:
        std_cell_info = None
        module_registry = None

    if netlist_sdf_file != '-':
        print('Reading SDF file... ')
        sdf_parser = SDFParser()
        sdf_header, sdf_cells = sdf_parser.read_file(netlist_sdf_file)
        print('Finished.')
    else:
        sdf_header, sdf_cells = None, None

    if netlist_gv_file != '-':
        print('Reading GV file... ', end='', flush=True)
        gv_info = GVParser.read_file(netlist_gv_file)
        print('Finished.')
        circuit = Circuit(gv_info, std_cell_info)
        circuit.summary()
    else:
        gv_info = None
        circuit = None

    with open(output_file, 'wb') as f_out:
        pickle.dump({
                'design_name': gv_info.id,
                'circuit': circuit,
                'module_registry': module_registry,
                'sdf_header': sdf_header,
                'sdf_cells': sdf_cells,
            }, f_out
        )
