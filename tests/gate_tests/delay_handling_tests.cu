#include "gtest/gtest.h"
#include <vector>

#include "simulator/simulator.h"
#include "simulator/data_structures.h"

using namespace std;

struct TestPair{
    vector<Transition> waveform;
    vector<DelayInfo> delay_infos;
    vector<Transition> expected;
    vector<char> edge_types;
    vector<unsigned> input_indices, output_indices;
    vector<int> rising_delays, falling_delays;
    unsigned int capacity = 16;
};

class DelayTestFixture: public ::testing::TestWithParam<TestPair>{
protected:

};

TEST_P(DelayTestFixture, SimpleCases) {
    const auto params = GetParam();
    const auto& capacity = params.capacity;
    auto waveform = params.waveform; waveform.resize(capacity);
    auto delay_infos = params.delay_infos; delay_infos.resize(capacity);

    auto** transitions = new Transition*;
    transitions[0] = new Transition[waveform.size()];
    for (int i = 0; i < capacity; i++) transitions[0][i] = waveform[i];

    unsigned int lengths = 0;

    unsigned int num_rows = params.edge_types.size();
    auto* edge_types = new char[num_rows];
    auto* input_indices = new unsigned int[num_rows];
    auto* output_indices = new unsigned int[num_rows];
    auto* rising_delays = new int[num_rows];
    auto* falling_delays = new int[num_rows];
    for (int i = 0; i < num_rows; i++) {
        edge_types[i] = params.edge_types[i];
        input_indices[i] = params.input_indices[i]; output_indices[i] = params.output_indices[i];
        rising_delays[i] = params.rising_delays[i]; falling_delays[i] = params.falling_delays[i];
    }
    SDFSpec sdf_spec = {
        .num_rows = num_rows,
        .input_index = input_indices,
        .output_index = output_indices,
        .edge_type = edge_types,
        .rising_delay = rising_delays,
        .falling_delay = falling_delays
    };

    compute_delay(transitions, capacity, delay_infos.data(), 1, 0, &sdf_spec, &lengths);

    int err_num = 0;
    const auto& expected = params.expected;
    EXPECT_EQ(lengths, expected.size());

    for (int i = 0; i < expected.size(); i++) {
        if (transitions[0][i] != expected[i]) err_num++;
    }

    delete[] transitions[0];
    delete transitions;
    EXPECT_EQ(err_num, 0);
}

INSTANTIATE_TEST_CASE_P(
    DelayTests,
    DelayTestFixture,
    ::testing::Values(
        TestPair{
            vector<Transition>{
                Transition{0, '1'},
                Transition{29091, '0'},
            },
            vector<DelayInfo>{ {}, {2, '+'} },
            vector<Transition>{
                Transition{29119, '0'},
            },
            vector<char>{'x', 'x', 'x'},
            vector<unsigned int>{0, 1, 2}, vector<unsigned int>{0, 0, 0},
            vector<int>{29, 44, 28}, vector<int>{29, 29, 28}
        },
        TestPair{
            vector<Transition>{
                Transition{0, '1'},
                Transition{21000, '0'}, Transition{21000, '0'},
            },
            vector<DelayInfo>{ {}, DelayInfo{0, '-'},  DelayInfo{1, '-'} },
            vector<Transition>{
                Transition{21015, '0'},
            },
            vector<char>{'x', 'x'},
            vector<unsigned int>{0, 1}, vector<unsigned int>{0, 0},
            vector<int>{38, 22}, vector<int>{25, 15}
        },
        TestPair{
            vector<Transition>{
                Transition{0, '0'},
                Transition{22000, '1'}, Transition{22000, '1'},
            },
            vector<DelayInfo>{ {}, DelayInfo{0, '+'}, DelayInfo{1, '+'} },
            vector<Transition>{
                Transition{22022, '1'},
            },
            vector<char>{'x', 'x'},
            vector<unsigned int>{0, 1}, vector<unsigned int>{0, 0},
            vector<int>{38, 22}, vector<int>{25, 15}
        },
        TestPair{
            vector<Transition>{
                Transition{50, '1'}, Transition{60, '0'},
            },
            vector<DelayInfo>{ DelayInfo{1, '+'}, DelayInfo{0, '-'} },
            vector<Transition>{
                Transition{65, '0'},
            },
            vector<char>{'x', 'x'},
            vector<unsigned int>{0, 1}, vector<unsigned int>{0, 0},
            vector<int>{8, 32}, vector<int>{5, 35}
        },
        TestPair{
            vector<Transition>{
                Transition{0, '0'},
                Transition{22091, '1'}, Transition{22091, '1'}, Transition{22091, '1'},
            },
            vector<DelayInfo>{ {}, DelayInfo{0, '+'}, DelayInfo{1, '-'}, DelayInfo{2, '-'} },
            vector<Transition>{
                Transition{22095, '1'}
            },
            vector<char>{'x', 'x', 'x'},
            vector<unsigned int>{0, 1, 2}, vector<unsigned int>{0, 0, 0},
            vector<int>{4, 18, 20}, vector<int>{3, 18, 20}
        }
    )
);
