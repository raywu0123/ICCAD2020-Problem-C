#ifndef ICCAD2020_CONSTANTS_H
#define ICCAD2020_CONSTANTS_H

#include <vector>

using namespace std;


typedef pair<int, int> BitWidth;
typedef pair<string, BitWidth> Multibit;
typedef vector<Multibit> ArgumentList;
typedef pair<string, ArgumentList> ArgumentPair;

typedef long long int Timestamp;

enum GV_IO_TYPE{
    GV_INPUT,
    GV_OUTPUT,
    GV_WIRE,
    Last=GV_WIRE,
};
const GV_IO_TYPE GV_IO_TYPES[] = {GV_INPUT, GV_OUTPUT, GV_WIRE};

const int NUM_VALUES = 4;
const char VALUES[NUM_VALUES] = {'0', '1', 'x', 'z'};

enum STD_CELL_DECLARE_TYPE {
    STD_CELL_INPUT,
    STD_CELL_OUTPUT,
    STD_CELL_WIRE,
    STD_CELL_SUPPLY1,
    STD_CELL_SUPPLY0,
    STD_CELL_LAST=STD_CELL_SUPPLY0,
};

const STD_CELL_DECLARE_TYPE STD_CELL_DECLARE_TYPES[] = {
    STD_CELL_INPUT, STD_CELL_OUTPUT, STD_CELL_WIRE, STD_CELL_SUPPLY1, STD_CELL_SUPPLY0
};

#endif //ICCAD2020_CONSTANTS_H
