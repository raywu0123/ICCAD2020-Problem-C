#ifndef ICCAD2020_UTILS_H
#define ICCAD2020_UTILS_H

#include <string>
#include <simulator/data_structures.h>

double get_timescale(int num, const std::string& unit);

template<class T>
std::vector<std::vector<T>> split_vector(const std::vector<T>& layer, unsigned int num_split) {
    std::vector<std::vector<T>> splits; splits.resize(num_split);
    int split_size = ceil(double(layer.size()) / double(num_split));
    for (int i = 0; i < num_split; ++i) {
        splits[i].reserve(split_size);
        for (int j = 0; j < split_size; ++j) {
            if (i * split_size + j >= layer.size()) break;
            splits[i].push_back(layer[i * split_size + j]);
        }
    }
    return splits;
}

void cudaErrorCheck(cudaError_t);
#endif
