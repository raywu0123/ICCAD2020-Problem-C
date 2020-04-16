#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "simulator/builtin_gates.h"

using namespace std;

TEST(gate_test, and_gate) {
    vector<vector<Transition>> inputs {
        { Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'z'} },
        { Transition{0, '0'}, Transition{2, '1'}, Transition{3, '1'}, Transition{4, '1'} },
        { Transition{0, '1'}, Transition{5, '0'}, Transition{6, 'z'}, Transition{7, 'x'}}
    };
    vector<Transition> expected_output {
        Transition{1, '0'}, Transition{2, '0'}, Transition{2, 'x'}, Transition{3, 'x'}
    };

    auto** data_schedule = new Transition*[inputs.size() + 1];
    vector<unsigned int> capacities;
    for (int i = 0; i < inputs.size(); i++) {
        data_schedule[i] = &(inputs[i][0]);
        capacities.push_back(inputs[i].size());
    }
    vector<Transition> output;
    output.insert(output.begin(), inputs[0].begin(), inputs[0].end());
    capacities.push_back(output.size());
    data_schedule[inputs.size()] = &output[0];
    and_gate_fn(data_schedule, &(capacities[0]), nullptr, inputs.size(), 1);

    int error_num = 0;
    for (int i = 0; i < output.size(); i++) {
        if (   output[i].timestamp != expected_output[i].timestamp
            or output[i].value != expected_output[i].value
        )
            error_num++;
    }
    EXPECT_EQ(error_num, 0);
}
