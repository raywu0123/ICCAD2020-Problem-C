#include <iostream>
#include "intermediate_file_reader.h"

#include "../specs/standard_cell.h"
#include "../specs/sdf_cell.h"


using namespace std;


void IntermediateFileReader::summary() const {
    cout << "Summary of intermediate file" << endl;
    cout << "Number of standard cells: " << StandardCellLibrary::cells.size() << endl;

    cout << "Number of gv io: " << endl;
    cout << "Number of multibit gv io: " << endl;

    cout << "SDF Timescale: " << timing_spec.timescale_num << " " << timing_spec.timescale_unit << endl;
    cout << "Number of sdf cells: " << timing_spec.cells.size() << endl;
    cout << endl;
}

void IntermediateFileReader::read(char* path) {
    cout << "Reading Intermediate File: " << path << " ..." << endl;
    fin = ifstream(path);
    read_vlib();
    return;
    circuit.read_file(fin);
    read_sdf();
}

void IntermediateFileReader::read_vlib() {
    int num_primitive_cells, num_modules;
    fin >> num_primitive_cells >> num_modules;

    for (int i = 0; i < num_primitive_cells; i++) {
        read_vlib_primitive();
    }
    for (int i = 0; i < num_modules; i++) {
        read_vlib_module();
    }
}

void IntermediateFileReader::read_vlib_primitive() {
    vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>> declares;
    string name = read_vlib_common(declares);
    vector<string> table;
    read_vlib_table(table);
    StandardCellLibrary::cells.emplace(name, new PrimitiveCell(declares, table));
}

void IntermediateFileReader::read_vlib_table(vector<string>& table) {
    int num_rows;
    fin >> num_rows;
    string row;
    for (int i = 0; i < num_rows; i++) {
        fin >> row;
        table.push_back(row);
    }
}

void IntermediateFileReader::read_vlib_module() {
    vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>> declares;
    string name = read_vlib_common(declares);
    int num_submodules;

    vector<pair<string, vector<string>>> submodules;
    fin >> num_submodules;
    for (int i = 0; i < num_submodules; i++) {
        string submod_type, submod_id, arg;
        int num_args;
        vector<string> args;
        fin >> submod_type >> submod_type >> submod_id >> num_args;
        for (int i_arg = 0; i_arg < num_args; i_arg++) {
            fin >> arg;
            args.push_back(arg);
        }
        submodules.emplace_back(submod_type, args);
    }
    StandardCellLibrary::cells.emplace(name, new Module(declares, submodules));
}

string IntermediateFileReader::read_vlib_common(vector<pair<STD_CELL_DECLARE_TYPE, vector<string>>>& declares) {
    string name;
    fin >> name;
    for (const auto& declare_type : STD_CELL_DECLARE_TYPES) {
        int num_args;
        string s;
        fin >> s >> num_args;
        vector<string> args;
        for (int i = 0; i < num_args; i++) {
            fin >> s;
            args.push_back(s);
        }
        declares.emplace_back(declare_type, args);
    }
    return name;
}

//void IntermediateFileReader::read_gv() {
//    fin >> g.design_name;
//    read_gv_io();
//    read_gv_assign();
//    read_gv_cells();
//}
//
//void IntermediateFileReader::read_gv_io() {
//    string s;
//    int num_args;
//    for (auto io_type : GV_IO_TYPES) {
//        fin >> s >> num_args;
//        vector<string> args;
//        for(int i_arg = 0; i_arg < num_args; i_arg++) {
//            fin >> s;
//            args.push_back(s);
//        }
//        g.gv_io[io_type] = args;
//
//        fin >> s >> num_args;
//        vector<Multibit> multibit_args;
//        for(int i_arg = 0; i_arg < num_args; i_arg++) {
//            BitWidth bitwidth;
//            fin >> s >> bitwidth.first >> bitwidth.second;
//            multibit_args.emplace_back(s, bitwidth);
//        }
//        g.gv_multibit_io[io_type] = multibit_args;
//    }
//}
//
//void IntermediateFileReader::read_gv_assign() {
//    int num_assigns;
//    fin >> num_assigns;
//    string lhs, rhs;
//    for (int i = 0; i < num_assigns; i++) {
//        BitWidth bitwidth;
//        fin >> lhs >> bitwidth.first >> bitwidth.second >> rhs;
//        auto lhs_with_bitwidth = make_pair(lhs, bitwidth);
//        g.gv_assign.emplace_back(lhs_with_bitwidth, rhs);
//    }
//}
//
//void IntermediateFileReader::read_gv_cells() {
//    int num_cells, num_params;
//    fin >> num_cells;
//    for (int i_cell = 0; i_cell < num_cells; i_cell++) {
//        string cell_type, cell_id;
//        fin >> cell_type >> cell_id >> num_params;
//        vector<ArgumentPair> params;
//        for(int i_param = 0; i_param < num_params; i_param++) {
//            ArgumentList args;
//            int num_args;
//            string pin_name;
//            fin >> pin_name >> num_args;
//            for(int i_arg = 0; i_arg < num_args; i_arg++) {
//                string arg;
//                BitWidth bitwidth;
//                fin >> arg >> bitwidth.first >> bitwidth.second;
//                args.emplace_back(arg, bitwidth);
//            }
//            params.emplace_back(pin_name, args);
//        }
//        g.gv_cells.emplace_back(new GVCell(cell_type, cell_id, params));
//    }
//}

void IntermediateFileReader::read_sdf() {
    string s;
    fin >> s >> timing_spec.timescale_num >> timing_spec.timescale_unit;
    int num_cells;
    fin >> num_cells;
    for (int i_cell = 0; i_cell < num_cells; i_cell++) {
        string type, name;
        int num_paths;
        fin >> type >> name >> num_paths;
        vector<pair<string, vector<string>>> paths;
        for(int i_path = 0; i_path < num_paths; i_path++) {
            string in;
            int num_out;
            fin >> in >> num_out;
            vector<string> outs;
            for(int i_out = 0; i_out < num_out; i_out++) {
                string out;
                fin >> out;
                outs.push_back(out);
            }
            paths.emplace_back(in, outs);
        }
        timing_spec.cells.emplace_back(new SDFCell(type, name, paths));
    }
}
