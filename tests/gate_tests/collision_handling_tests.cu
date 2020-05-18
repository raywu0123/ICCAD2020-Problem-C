#include "gtest/gtest.h"
#include <vector>

#include "simulator/simulator.h"

using namespace std;

struct TestPair{
    vector<Transition> waveform;
    vector<Transition> expected;
};

class SingleWaveformCollisionTestFixture: public ::testing::TestWithParam<TestPair>
{
protected:
};


TEST_P(SingleWaveformCollisionTestFixture, SimpleCases) {
    auto& test_pair = GetParam();
    auto waveform = test_pair.waveform;

    unsigned int length = 0;
    resolve_collisions_for_single_waveform(waveform.data(), waveform.size(), &length);

    EXPECT_EQ(length, test_pair.expected.size());
    unsigned int err_num = 0;

    for (int i = 0; i < test_pair.expected.size(); i++) {
        if (waveform[i] != test_pair.expected[i]) err_num++;
    }
    EXPECT_EQ(err_num, 0);
}

INSTANTIATE_TEST_CASE_P(
    GateTests,
    SingleWaveformCollisionTestFixture,
    ::testing::Values(
        TestPair{
            vector<Transition>{
                Transition{0, 0}, Transition{3, 0}, Transition{5, 0}, Transition{4, 0}, Transition{6, 0}
            },
            vector<Transition>{
                Transition{0, 0}, Transition{3, 0}, Transition{4, 0}, Transition{6, 0}
            }
        },
        TestPair{
            vector<Transition>{
                Transition{0, 0}, Transition{5, 'x'}, Transition{5, 'y'}
            },
            vector<Transition>{
                Transition{0, 0}, Transition{5, 'y'}
            }
        },
        TestPair{
            vector<Transition>{
                Transition{0, 0}, Transition{5, 0}, Transition{4, 0}, Transition{6, 0}, Transition{3, 0}
            },
            vector<Transition>{
                Transition{0, 0}, Transition{3, 0}
            }
        },
        TestPair{
            vector<Transition>{
                Transition{10, 0}, Transition{11, 0}, Transition{0, 0}, Transition{5, 0}
            },
            vector<Transition>{
                Transition{0, 0}, Transition{5, 0}
            }
        }
    )
);