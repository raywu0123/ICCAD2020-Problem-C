#ifndef ICCAD2020_CIRCUIT_H
#define ICCAD2020_CIRCUIT_H

#include <fstream>
#include <string>


using namespace std;

class Circuit {
public:
    void read_file(ifstream& fin) {};
    void summary() const {};
};

#endif //ICCAD2020_CIRCUIT_H