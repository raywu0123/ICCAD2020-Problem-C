#include <iostream>
#include "intermediate_file_reader.h"

#include "../specs/standard_cell.h"


using namespace std;


void IntermediateFileReader::summary() const {
    cout << "Summary of intermediate file" << endl;
    cout << "Number of standard cells: " << StandardCellLibrary::cells.size() << endl;
    cout << "SDF Timescale: " << timing_spec.timescale_num << " " << timing_spec.timescale_unit << endl;
    cout << "Number of sdf cells: " << timing_spec.cells.size() << endl;
    cout << endl;
}

void IntermediateFileReader::read(char* path) {
    cout << "Reading Intermediate File: " << path << " ..." << endl;
    fin = ifstream(path);
    read_vlib();
    circuit.read_file(fin);
    return;
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
