from sys import argv

from FileParsers import read_vlib_file, SDFParser


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

    print('Reading standard cell library... ', end='', flush=True)
    std_cell_info = read_vlib_file(std_cells_file)
    print('Finished.')

    print('Reading SDF file... ', end='', flush=True)
    sdf_parser = SDFParser()
    sdf_info = sdf_parser.read_file(netlist_sdf_file)
    print('Finished.')

