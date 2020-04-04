from sys import argv

from graph_preprocessing.file_parsers import VlibReader, SDFParser, GVParser
from graph_preprocessing.intermediate_file_writer import IntermediateFileWriter

from graph_preprocessing.circuit_model.circuit_model import Circuit
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
    else:
        std_cell_info = None

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
    else:
        gv_info = None

    if gv_info is not None:
        circuit = Circuit(gv_info, std_cell_info)
        circuit.summary()
    else:
        circuit = None

    with IntermediateFileWriter(output_file) as writer:
        if std_cell_info is not None:
            writer.write_vlib(std_cell_info)

        if circuit is not None:
            writer.write_graph(circuit.graph)

        if sdf_header is not None and sdf_cells is not None:
            writer.write_sdf(sdf_header, sdf_cells)

