//
// Created by ray on 3/30/20.
//

#ifndef ICCAD2020_GV_CELL_H
#define ICCAD2020_GV_CELL_H

#include <string>
#include <utility>
#include <vector>

#include "../constants.h"


using namespace std;
class GVCell {
public:
    GVCell(string type, string id): type(std::move(type)), id(std::move(id)) {};

    string type, id;
};


#endif //ICCAD2020_GV_CELL_H
