#ifndef ICCAD2020_GRAPH_H
#define ICCAD2020_GRAPH_H

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "input_waveforms.h"
#include "specs/gv_cell.h"
#include "specs/standard_cell.h"
#include "constants.h"
#include "data_containers.h"

using namespace std;

class Node {
    vector<shareshared_ptr<SlicedWire>> input_edges;
    vector<shareshared_ptr<SlicedWire>> output_edges;
};

class SlicedWire {
//    edges on the graph
public:
    SlicedWire(const Wire& original_wire, Bitwidth bitwidth):
        original_wire(original_wire), bitwidth(std::move(bitwidth)) {
//        TODO
    };

    vector<shared_ptr<Node>> input_nodes;
    vector<shared_ptr<Node>> output_nodes;

    Wire& original_wire;
    BitWidth bitwidth;
};

class Wire: public Node {
//    owns data
public:
    const shared_ptr<SlicedWire> slice(Bitwidth bitwidth) {
//        TODO
    }
};

class Cell: public Node {
public:
    void compute();
};

class PlaceHolder: public Wire {
public:
    void set_data();
};


class Graph {
public:
    Graph() = default;

    void summary() const;
    void read_file(ifstream& fin);
    bool verify(const InputWaveforms&);

    string design_name;
};

#endif //ICCAD2020_GRAPH_H