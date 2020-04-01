#ifndef ICCAD2020_SDF_CELL_H
#define ICCAD2020_SDF_CELL_H

#include <string>
#include <vector>

using namespace std;

class SDFCell {
public:
    SDFCell(const string& type, const string& id, const vector<pair<string, vector<string>>>& paths);
};


#endif //ICCAD2020_SDF_CELL_H
