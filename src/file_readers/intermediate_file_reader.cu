#include <iostream>
#include "intermediate_file_reader.h"


using namespace std;


void IntermediateFileReader::summary() const {
    cout << "Summary of intermediate file" << endl;
    cout << "SDF Timescale: " << timing_spec.timescale_num << " " << timing_spec.timescale_unit << endl;
    cout << "Number of sdf cells: " << timing_spec.cells.size() << endl;
    cout << endl;
}

void IntermediateFileReader::read(char* path) {
    cout << "Reading Intermediate File: " << path << " ..." << endl;
    fin = ifstream(path);
    read_vlib();
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
    StdCellDeclare declares;
    string name = read_vlib_common(declares);
    vector<string> table;
    read_vlib_table(table);
    module_registry.register_user_defined_primitive(
            name,
            table,
            declares.buckets[STD_CELL_INPUT].size(),
            declares.buckets[STD_CELL_OUTPUT].size()
    );
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
    StdCellDeclare declares;
    string name = read_vlib_common(declares);
    int num_submodules;

    unordered_map<string, int> arg_name_to_index;
    for (const auto& bucket : declares.buckets) {
        for (const auto &arg_name: bucket) {
            arg_name_to_index[arg_name] = arg_name_to_index.size();
        }
    }

    vector<pair<string, vector<int>>> submodules;
    fin >> num_submodules;
    for (int i = 0; i < num_submodules; i++) {
        string submod_type, submod_id, arg;
        int num_args;
        vector<int> args;
        fin >> submod_type >> submod_type >> submod_id >> num_args;
        for (int i_arg = 0; i_arg < num_args; i_arg++) {
            fin >> arg;
            args.push_back(arg_name_to_index[arg]);
        }
        submodules.emplace_back(submod_type, args);
    }
    module_registry.register_module(name, submodules);
}

string IntermediateFileReader::read_vlib_common(StdCellDeclare& declares) {
    string name;
    fin >> name;
    for (auto& arg_bucket : declares.buckets) {
        int num_args;
        string s;
        fin >> s >> num_args;
        for (int i = 0; i < num_args; i++) {
            fin >> s;
            arg_bucket.push_back(s);
        }
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
