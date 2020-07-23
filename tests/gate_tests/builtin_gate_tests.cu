#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "simulator/builtin_gates.h"
#include "constants.h"

using namespace std;

struct SingleWaveformTestPair {
    GateFnPtr gate_fn;
    vector<Transition> expected_output;
};

class BuiltinGateTestFixture: public ::testing::TestWithParam<SingleWaveformTestPair>
{
public:
    BuiltinGateTestFixture() {
        for (auto& inp : inputs) { inp.resize(INITIAL_CAPACITY); }
    }
protected:
    vector<vector<Transition>> inputs {
        { Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'z'} },
        { Transition{0, '0'}, Transition{1, '1'}, Transition{2, '1'}, Transition{3, '1'} },
        { Transition{0, '1'}, Transition{1, '0'}, Transition{2, 'z'}, Transition{3, 'x'} }
    };
};

TEST_P(BuiltinGateTestFixture, SimpleCases) {
    auto test_pair = GetParam();
    auto gate_fn = test_pair.gate_fn;
    auto expected_output = test_pair.expected_output;

    vector<Transition> output; output.resize(INITIAL_CAPACITY);

    auto** data_schedule = new Transition*[inputs.size() + 1];
    data_schedule[0] = output.data();
    for (int i = 0; i < inputs.size(); i++) {
        data_schedule[i + 1] = inputs[i].data();
    }
    gate_fn(data_schedule, nullptr, 0, inputs.size(), 1);

    int error_num = 0;
    for (int i = 0; i < expected_output.size(); i++) {
        if (   output[i].timestamp != expected_output[i].timestamp
            or output[i].value != expected_output[i].value
        ) {
            error_num++;
        }
    }
    EXPECT_EQ(error_num, 0);
    delete[] data_schedule;
}


INSTANTIATE_TEST_SUITE_P(
    GateTests,
    BuiltinGateTestFixture,
    ::testing::Values(
            SingleWaveformTestPair{
            and_gate_fn,
            vector<Transition>{ Transition{0, '0'}, Transition{1, '0'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
            SingleWaveformTestPair{
            or_gate_fn,
            vector<Transition>{ Transition{0, '1'}, Transition{1, '1'}, Transition{2, '1'}, Transition{3, '1'} }
        },
            SingleWaveformTestPair{
            xor_gate_fn,
            vector<Transition>{ Transition{0, '1'}, Transition{1, '0'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
            SingleWaveformTestPair{
            nand_gate_fn,
            vector<Transition>{ Transition{0, '1'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
            SingleWaveformTestPair{
            nor_gate_fn,
            vector<Transition>{ Transition{0, '0'}, Transition{1, '0'}, Transition{2, '0'}, Transition{3, '0'} }
        },
            SingleWaveformTestPair{
            xnor_gate_fn,
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
            SingleWaveformTestPair{
            not_gate_fn,
            vector<Transition>{ Transition{0, '1'}, Transition{1, '0'}, Transition{2, 'x'}, Transition{3, 'x'} }
        },
            SingleWaveformTestPair{
            buf_gate_fn,
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'x'} }
        }
    )
);


struct PrimitiveTestPair {
    vector<string> table;
    vector<Transition> expected_output;
};

class PrimitiveGateTestFixture: public ::testing::TestWithParam<PrimitiveTestPair>
{
public:
    PrimitiveGateTestFixture() {
        for(auto& inp : inputs) { inp.resize(INITIAL_CAPACITY); }
    }
protected:
    vector<vector<Transition>> inputs {
        { Transition{0, '1'}, Transition{1, '0'}, Transition{2, '1'} },
        { Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'} },
        { Transition{0, 'x'}, Transition{1, '0'}, Transition{2, '1'} }
    };
};

TEST_P(PrimitiveGateTestFixture, SimpleCases) {
    auto test_pair = GetParam();
    auto vector_table = test_pair.table;
    auto table_row_num = test_pair.table.size();
    char* table = new char[(inputs.size() + 1) * table_row_num];
    for (int i_table_row = 0; i_table_row < table_row_num; i_table_row++) {
        for (int i = 0; i < inputs.size() + 1; i++) {
            table[i_table_row * (inputs.size() + 1) + i] = vector_table[i_table_row][i];
        }
    }

    auto expected_output = test_pair.expected_output; expected_output.resize(INITIAL_CAPACITY);

    vector<Transition> output;
    output.resize(INITIAL_CAPACITY);
    auto** data_schedule = new Transition*[inputs.size() + 1];
    data_schedule[0] = output.data();
    for (int i = 0; i < inputs.size(); i++) {
        data_schedule[i + 1] = inputs[i].data();
    }
    primitive_gate_fn(data_schedule, table, table_row_num, inputs.size(), 1);

    int error_num = 0;
    for (int i = 0; i < expected_output.size(); i++) {
        if (output[i] != expected_output[i]) error_num++;
    }
    EXPECT_EQ(error_num, 0);
    delete[] table;
}


INSTANTIATE_TEST_SUITE_P(
    GateTests,
    PrimitiveGateTestFixture,
    ::testing::Values(
        PrimitiveTestPair{
            vector<string>{"1?01", "0?00", "?111", "?010", "00x0", "11x1"},
            vector<Transition>{
                Transition{0, 'x'}, Transition{1, '0'}, Transition{2, 'x'},
            }
        }
    )
);

