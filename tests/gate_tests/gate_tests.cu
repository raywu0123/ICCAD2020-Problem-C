#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "simulator/builtin_gates.h"

using namespace std;

TEST(gate_test, and_gate) {
    GateFnPtr and_gate_fn;
    vector<vector<Transition>> inputs {
        { Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'z'} },
        { Transition{1, '0'}, Transition{2, '1'}, Transition{3, '1'}, Transition{4, '1'} },
        { Transition{0, '1'}, Transition{5, '0'}, Transition{6, 'z'}, Transition{7, 'x'}}
    };

    auto** input_data = new Transition*[inputs.size()];
    vector<unsigned int> capacities;
    for (int i = 0; i < inputs.size(); i++) {
        input_data[i] = &(inputs[i][0]);
        capacities.push_back(inputs[i].size());
    }
    vector<Transition> output;
    output.insert(output.begin(), inputs[0].begin(), inputs[0].end());
    capacities.push_back(output.size());
    (*and_gate_fn_ptr)(input_data, &(capacities[0]), nullptr, inputs.size(), 1);

    int error_num = 0;
    EXPECT_EQ(error_num, 0);
}
