#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "simulator/simulator.h"

using namespace std;

struct TestData {
    vector<vector<Transition>> inputs;
    vector<vector<Transition>> expected_outputs;
    vector<DelayInfo> expected_delay_infos;
    bool expected_overflow;
    CAPACITY_TYPE capacity = 4;
};

class SteppingTestFixture: public ::testing::TestWithParam<TestData>{
protected:
};

TEST_P(SteppingTestFixture, SimpleCases) {
    auto test_data = GetParam();
    auto& inputs = test_data.inputs;
    const auto& capacity = test_data.capacity;

    Values table[16 * 2] = {
        Values::ZERO, Values::ZERO,
        Values::ZERO, Values::ONE,
        Values::ZERO, Values::X,
        Values::ZERO, Values::Z,
        Values::ONE, Values::ZERO,
        Values::ONE, Values::ONE,
        Values::ONE, Values::X,
        Values::ONE, Values::Z,
        Values::X, Values::ZERO,
        Values::X, Values::ONE,
        Values::X, Values::X,
        Values::X, Values::Z,
        Values::Z, Values::ZERO,
        Values::Z, Values::ONE,
        Values::Z, Values::X,
        Values::Z, Values::Z,
    };
    ModuleSpec module_spec{.num_input = 2, .num_output = 2, .table = table};

    Transition* input_data[MAX_NUM_MODULE_OUTPUT] = {nullptr};
    unsigned int sizes[MAX_NUM_MODULE_OUTPUT] = {0};
    for (int i = 0; i < inputs.size(); ++i) {
        input_data[i] = inputs[i].data();
        sizes[i] = inputs[i].size();
    }
    Transition* output_data[MAX_NUM_MODULE_OUTPUT] = {nullptr};
    for (int i = 0; i < module_spec.num_output; ++i) output_data[i] = new Transition[capacity];
    DelayInfo* s_delay_infos = new DelayInfo[capacity];

    bool overflow = false;
    stepping_algorithm(
        input_data, sizes,
        s_delay_infos,
        output_data,
        &module_spec,
        capacity,
        &overflow
    );
    for (int i = 0; i < module_spec.num_output; ++i) {
        for (int j = 0; j < capacity; ++j) {
            cout << output_data[i][j] << "";
        }
        cout << endl;
    }

    ASSERT_EQ(test_data.expected_overflow, overflow);

    if (not overflow) {
        int num_output_err = 0;
        for (int o = 0; o < module_spec.num_output; ++o) {
            const auto& expected_output = test_data.expected_outputs[o];
            for (int i = 0; i < capacity; ++i) {
                if (i < expected_output.size()) {
                    if (output_data[o][i] != expected_output[i]) num_output_err++;
                } else if (output_data[o][i] != Transition{0, 0}) num_output_err++;
            }
        }
        ASSERT_EQ(num_output_err, 0);

        int num_delay_info_err = 0;
        const auto& expected_delay_infos = test_data.expected_delay_infos;
        for (int i = 0; i < capacity; ++i) {
            if (i < expected_delay_infos.size()) {
                if (not (s_delay_infos[i] == expected_delay_infos[i])) num_delay_info_err++;
            } else if (not (s_delay_infos[i] == DelayInfo{}))
                num_delay_info_err++;
        }
    }

    for (int i = 0; i < module_spec.num_output; ++i) delete[] output_data[i];
    delete[] s_delay_infos;
}

INSTANTIATE_TEST_CASE_P(
    SteppingTests,
    SteppingTestFixture,
    ::testing::Values(
        TestData{
            vector<vector<Transition>> {
                { Transition{0, '1'}, Transition{10, '0'}, Transition{20, 'x'} },
                { Transition{1, '0'}, Transition{10, 'x'} }
            },
            vector<vector<Transition>> {
                { Transition{0, '1'}, Transition{10, '0'}, Transition{10, '0'}, Transition{20, 'x'} },
                { Transition{0, '0'}, Transition{10, 'x'}, Transition{10, 'x'}, Transition{20, 'x'} }
            },
            vector<DelayInfo> {
                {0, EdgeTypes::NODELAY}, {0, EdgeTypes::FALLING}, {1, EdgeTypes::RISING}, {0, EdgeTypes::RISING}
            },
            false,
        },
        TestData{
            vector<vector<Transition>> {
                { Transition{0, '1'}, Transition{10, '0'}, Transition{20, 'x'} },
                { Transition{0, '0'}, Transition{10, 'x'} }
            },
            vector<vector<Transition>> {
                { Transition{ULONG_MAX, '1'}, Transition{10, '0'}, Transition{10, '0'}, Transition{20, 'x'} },
                { Transition{ULONG_MAX, '0'}, Transition{10, 'x'}, Transition{10, 'x'}, Transition{20, 'x'} }
            },
            vector<DelayInfo> {
                {0, EdgeTypes::NODELAY}, {0, EdgeTypes::FALLING}, {1, EdgeTypes::RISING}, {0, EdgeTypes::RISING}
            },
            false,
        },
        TestData{
            vector<vector<Transition>> {
                { Transition{0, '1'}, Transition{15, '0'}, Transition{20, 'x'}, Transition{30, '1'}, Transition{40, '0'},
                    Transition{50, '1'}, Transition{60, '0'}, Transition{70, '1'}, Transition{80, '0'}, Transition{ 90, '1'} },
                { Transition{1, '0'}, Transition{10, 'x'}, Transition{50, '1'}, Transition{55, '0'}, Transition{65, '1'},
                    Transition{75, '0'}, Transition{85, '1'}, Transition{95, '0'}, Transition{105, '1'}, Transition{115, '0'} }
            },
            vector<vector<Transition>> {},
            vector<DelayInfo> {},
            true,
        }
    )
);