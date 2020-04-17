#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "simulator/builtin_gates.h"

using namespace std;

struct TestPair {
    GateFnPtr gate_fn;
    vector<Transition> expected_output;
};

class BuiltinGateTestFixture: public ::testing::TestWithParam<TestPair>
{
protected:
    vector<vector<Transition>> inputs {
        { Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'z'} },
        { Transition{0, '0'}, Transition{2, '1'}, Transition{3, '1'}, Transition{4, '1'} },
        { Transition{0, '1'}, Transition{5, '0'}, Transition{6, 'z'}, Transition{7, 'x'}}
    };
};

TEST_P(BuiltinGateTestFixture, SimpleCases) {
    auto test_pair = GetParam();
    auto gate_fn = test_pair.gate_fn;
    auto expected_output = test_pair.expected_output;

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
    gate_fn(data_schedule, &(capacities[0]), nullptr, inputs.size(), 1);

    int error_num = 0;
    for (int i = 0; i < expected_output.size(); i++) {
        if (   output[i].timestamp != expected_output[i].timestamp
            or output[i].value != expected_output[i].value
        )
            error_num++;
    }
    EXPECT_EQ(error_num, 0);
}


INSTANTIATE_TEST_SUITE_P(
    GateTests,
    BuiltinGateTestFixture,
    ::testing::Values(
        TestPair{
            and_gate_fn,
            vector<Transition>{ Transition{1, '0'}, Transition{2, '0'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
        TestPair{
            or_gate_fn,
            vector<Transition>{ Transition{1, '1'}, Transition{2, '1'}, Transition{2, '1'}, Transition{3, '1'} }
        },
        TestPair{
            xor_gate_fn,
            vector<Transition>{ Transition{1, '0'}, Transition{2, 'x'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
        TestPair{
            nand_gate_fn,
            vector<Transition>{ Transition{1, '1'}, Transition{2, '1'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
        TestPair{
            nor_gate_fn,
            vector<Transition>{ Transition{1, '0'}, Transition{2, '0'}, Transition{2, '0'}, Transition{3, '0'} }
        },
        TestPair{
            xnor_gate_fn,
            vector<Transition>{ Transition{1, '1'}, Transition{2, 'x'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
        TestPair{
            not_gate_fn,
            vector<Transition>{ Transition{1, '0'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
        TestPair{
            buf_gate_fn,
            vector<Transition>{ Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'x'} }
        }
    )
);
