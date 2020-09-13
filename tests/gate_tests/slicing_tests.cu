#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "simulator/simulator.h"

using namespace std;

struct TestData {
    vector<vector<Transition>> inputs;
    vector<vector<SliceInfo>> expected_s_slice_infos;
    unsigned int capacity = 4;
};

class SliceWaveformTestFixture: public ::testing::TestWithParam<TestData>{
protected:
};

TEST_P(SliceWaveformTestFixture, SimpleCases) {
    const auto test_data = GetParam();
    const auto& num_wires = test_data.inputs.size();
    const auto& capacity = test_data.capacity;
    SliceInfo s_slice_infos[(N_STIMULI_PARALLEL + 1) * MAX_NUM_MODULE_OUTPUT] = {0};

    auto inputs = test_data.inputs;
    InputData data[MAX_NUM_MODULE_ARGS];

    vector<Transition> input_data_collector;
    for (int i = 0; i < num_wires; ++i) {
        unsigned int offset = input_data_collector.size();
        input_data_collector.insert(input_data_collector.end(), inputs[i].begin(), inputs[i].end());
        data[i] = { offset, static_cast<unsigned int>(inputs[i].size()) };
    }

    bool overflow = false;
    auto* all_input_data = input_data_collector.data();
    slice_waveforms(s_slice_infos, all_input_data, data, capacity - 1, num_wires, &overflow);

    int err_num = 0;
    auto expected_slice_infos = test_data.expected_s_slice_infos;
    for (int i = 0; i < num_wires; ++i) {
        for (int j = 0; j < N_STIMULI_PARALLEL + 1; ++j) {
            if (j < expected_slice_infos[i].size()) {
                if (s_slice_infos[j * num_wires + i].offset != expected_slice_infos[i][j].offset) err_num++;
            } else if (s_slice_infos[j * num_wires + i].offset != 0) err_num++;
        }
    }

    ASSERT_EQ(err_num, 0);
}

INSTANTIATE_TEST_CASE_P(
    SliceWaveformTests,
    SliceWaveformTestFixture,
    ::testing::Values(
        TestData{
            vector<vector<Transition>> {
                { Transition{0, '1'}, Transition{10, '0'}, Transition{20, 'x'}, Transition{30, '1'} },
                { Transition{1, '0'}, Transition{10, 'x'}, Transition{50, '1'}, Transition{55, '0'} }
            },
            vector<vector<SliceInfo>> {
                {{0}, {3}, {4}},
                {{0}, {2}, {4}},
            }
        },
        TestData{
            vector<vector<Transition>> {
                { Transition{0, '1'}, Transition{15, '0'}, Transition{20, 'x'}, Transition{30, '1'}, Transition{40, '0'},
                  Transition{50, '1'}, Transition{60, '0'}, Transition{70, '1'}, Transition{80, '0'}, Transition{ 90, '1'} },
                { Transition{1, '0'}, Transition{10, 'x'}, Transition{50, '1'}, Transition{55, '0'}, Transition{65, '1'},
                  Transition{75, '0'}, Transition{85, '1'}, Transition{95, '0'}, Transition{105, '1'}, Transition{115, '0'} }
            },
            vector<vector<SliceInfo>> {
                {{0}, {3}, {6}, {8}, {10}, {10}},
                {{0}, {2}, {3}, {6}, {9}, {10}},
            }
        }
    )
);