#include <iostream>
#include "intermediate_file_reader.h"


using namespace std;


void IntermediateFileReader::summary() const {
    cout << "Summary of intermediate file" << endl;
    cout << "SDF Timescale: " << timing_spec.timescale_num << " " << timing_spec.timescale_unit << endl;
    cout << "Number of sdf cells: " << timing_spec.cells.size() << endl;
    cout << endl;
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
