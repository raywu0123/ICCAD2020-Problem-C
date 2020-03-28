from sys import argv

from graph_preprocessing.FileParsers import VlibReader, SDFParser, GVParser
from graph_preprocessing.intermediate_file_writer import IntermediateFileWriter


if __name__ == '__main__':
    if len(argv) != 5:
        raise ValueError(
            "Usage: FileParsers.cu.py "
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

    if netlist_sdf_file != '-':
        print('Reading SDF file... ')
        sdf_parser = SDFParser()
        sdf_header, sdf_cells = sdf_parser.read_file(netlist_sdf_file)
        print('Finished.')

    if netlist_gv_file != '-':
        print('Reading GV file... ', end='', flush=True)
        gv_info = GVParser.read_file(netlist_gv_file)
        print('Finished.')

    with IntermediateFileWriter(output_file) as writer:
        writer.write_vlib(std_cell_info)
        writer.write_gv(gv_info)
        writer.write_sdf(sdf_header, sdf_cells)

