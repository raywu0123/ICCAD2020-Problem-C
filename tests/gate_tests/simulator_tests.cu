#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "simulator/simulator.h"

using namespace std;

struct TestData {
    vector<vector<Transition>> inputs;
    vector<vector<char>> expected_s_values;
    vector<Timestamp> expected_s_timestamps;
    vector<DelayInfo> expected_s_delay_infos;
    unsigned int capacity = 16;
};

class SliceWaveformTestFixture: public ::testing::TestWithParam<TestData>{
protected:
};

TEST_P(SliceWaveformTestFixture, SimpleCases) {
    const auto test_data = GetParam();
    const auto& num_wires = test_data.inputs.size();
    const auto& capacity = test_data.capacity;
    auto* s_timestamps = new Timestamp[N_STIMULI_PARALLEL * capacity];
    auto* s_delay_infos = new DelayInfo[N_STIMULI_PARALLEL * capacity];
    auto* s_values = new Values[N_STIMULI_PARALLEL * capacity * num_wires];

    auto inputs = test_data.inputs;
    InputData data[MAX_NUM_MODULE_INPUT];

    vector<Transition> input_data_collector;
    for (int i = 0; i < num_wires; ++i) {
        unsigned int offset = input_data_collector.size();
        input_data_collector.insert(input_data_collector.end(), inputs[i].begin(), inputs[i].end());
        data[i] = { offset, static_cast<unsigned int>(inputs[i].size()) };
    }

    bool overflow = false;
    auto* all_input_data = input_data_collector.data();
    slice_waveforms(s_timestamps, s_delay_infos, s_values, all_input_data, data, 16, num_wires, &overflow);

    int timestamp_err_num = 0;
    auto expected_s_timestamps = test_data.expected_s_timestamps;
    expected_s_timestamps.resize(N_STIMULI_PARALLEL * capacity);
    for (int i = 0; i < N_STIMULI_PARALLEL * capacity; ++i) {
        if (s_timestamps[i] == expected_s_timestamps[i]) continue;
        timestamp_err_num++;
    }
    printf("\n");
    ASSERT_EQ(timestamp_err_num, 0);

    int delay_info_err_num = 0;
    auto expected_s_delay_infos = test_data.expected_s_delay_infos;
    expected_s_delay_infos.resize(N_STIMULI_PARALLEL * capacity);
    for (int i = 0; i < N_STIMULI_PARALLEL * capacity; ++i) {
        if (s_delay_infos[i] == expected_s_delay_infos[i]) continue;
        delay_info_err_num++;
    }
    ASSERT_EQ(delay_info_err_num, 0);

    int value_err_num = 0;
    auto expected_s_values = test_data.expected_s_values;
    for(auto& vs : expected_s_values) vs.resize(N_STIMULI_PARALLEL * capacity);
    for (int i = 0; i < num_wires; ++i) {
        for(int j = 0; j < N_STIMULI_PARALLEL * capacity; ++j) {
            if (s_values[j * num_wires + i] == raw_to_enum(expected_s_values[i][j])) continue;
            value_err_num++;
        }
    }
    ASSERT_EQ(value_err_num, 0);

    delete[] s_values, delete[] s_delay_infos, delete[] s_timestamps;
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
            vector<vector<char>> {
                {'1', '0', '0', 'x', '1', '1', '1'},
                {'0', 'x', 'x', 'x', 'x', '1', '0'}
            },
            vector<Timestamp> {0, 10, 10, 20, 30, 50, 55},
            vector<DelayInfo> { DelayInfo{}, DelayInfo{0, '-'}, DelayInfo{1, '+'}, DelayInfo{0, '+'}, DelayInfo{0, '+'}, DelayInfo{1, '+'}, DelayInfo{1, '-'} },
        },
        TestData{
            vector<vector<Transition>> {
                { Transition{0, '1'}, Transition{15, '0'}, Transition{20, 'x'}, Transition{30, '1'}, Transition{40, '0'},
                  Transition{50, '1'}, Transition{60, '0'}, Transition{70, '1'}, Transition{80, '0'}, Transition{ 90, '1'} },
                { Transition{1, '0'}, Transition{10, 'x'}, Transition{50, '1'}, Transition{55, '0'}, Transition{65, '1'},
                  Transition{75, '0'}, Transition{85, '1'}, Transition{95, '0'}, Transition{105, '1'}, Transition{115, '0'} }
            },
            vector<vector<char>> {
                {'1', '1', '0', 'x', '1', '0', '1', '1', '1', '0', '0', '1', '1', '0', '0', '1',
                 '1', '1', '1', '1'},
                {'0', 'x', 'x', 'x', 'x', 'x', '1', '1', '0', '0', '1', '1', '0', '0', '1', '1',
                 '1', '0', '1', '0'}
            },
            vector<Timestamp> {
                0, 10, 15, 20, 30, 40, 50, 50, 55, 60, 65, 70, 75, 80, 85, 90,
                90, 95, 105, 115
            },
            vector<DelayInfo> {
                DelayInfo{}, DelayInfo{1, '+'}, DelayInfo{0, '-'}, DelayInfo{0, '+'},
                DelayInfo{0, '+'}, DelayInfo{0, '-'}, DelayInfo{0, '+'}, DelayInfo{1, '+'},
                DelayInfo{1, '-'}, DelayInfo{0, '-'}, DelayInfo{1, '+'}, DelayInfo{0, '+'},
                DelayInfo{1, '-'}, DelayInfo{0, '-'}, DelayInfo{1, '+'}, DelayInfo{0, '+'},
                DelayInfo{}, DelayInfo{1, '-'}, DelayInfo{1, '+'}, DelayInfo{1, '-'}
            },
        }
    )
);