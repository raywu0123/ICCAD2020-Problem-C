#include "gtest/gtest.h"
#include <vector>

#include "simulator/simulator.h"
#include "simulator/data_structures.h"

using namespace std;

struct TestPair{
    vector<Transition> waveform;
    vector<DelayInfo> delay_infos;
    vector<Transition> expected;
    vector<SDFPath> sdf_paths;
    CAPACITY_TYPE capacity = 16;
};

class DelayTestFixture: public ::testing::TestWithParam<TestPair>{
protected:

};

TEST_P(DelayTestFixture, SimpleCases) {
    const auto params = GetParam();
    const auto& capacity = params.capacity;
    auto waveform = params.waveform; waveform.resize(capacity);
    auto delay_infos = params.delay_infos; delay_infos.resize(capacity);
    const auto sdf_paths = params.sdf_paths;

    auto** transitions = new Transition*;
    transitions[0] = new Transition[waveform.size()];
    for (int i = 0; i < capacity; i++) transitions[0][i] = waveform[i];

    CAPACITY_TYPE lengths = 0;

    compute_delay(transitions, capacity, delay_infos.data(), 1, 0, sdf_paths.data(), sdf_paths.size(), &lengths);

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
            vector<SDFPath> {
                {'x', 0, 1, 29, 29},
                {'x', 1, 0, 44, 29},
                {'x', 2, 0, 28, 28}
            }
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
            vector<SDFPath> {
                {'x', 0, 0, 38, 25},
                {'x', 1, 0, 22, 15}
            }
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
            vector<SDFPath> {
                {'x', 0, 0, 38, 25},
                {'x', 1, 0, 22, 15}
            }
        },
        TestPair{
            vector<Transition>{
                Transition{50, '1'}, Transition{60, '0'},
            },
            vector<DelayInfo>{ DelayInfo{1, '+'}, DelayInfo{0, '-'} },
            vector<Transition>{
                Transition{65, '0'},
            },
            vector<SDFPath> {
                {'x', 0, 0, 8, 5},
                {'x', 1, 0, 32, 35}
            }
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
            vector<SDFPath> {
                {'x', 0, 0, 4, 3},
                {'x', 1, 0, 18, 18},
                {'x', 2, 0, 20, 20}
            }
        }
    )
);
