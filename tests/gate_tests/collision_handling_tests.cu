#include "gtest/gtest.h"
#include <vector>

#include "simulator/simulator.h"

using namespace std;

struct SingleWaveformTestPair{
    vector<Transition> waveform;
    vector<Transition> expected;
};

class SingleWaveformCollisionTestFixture: public ::testing::TestWithParam<SingleWaveformTestPair>{};


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
        SingleWaveformTestPair{
            vector<Transition>{ Transition{}, Transition{0, '0'}, Transition{3, '0'}, Transition{5, '1'}, Transition{0, 0} },
            vector<Transition>{ Transition{0, '0'}, Transition{5, '1'} }
        },
        SingleWaveformTestPair{
            vector<Transition>{ Transition{}, Transition{0, '0'}, Transition{3, '1'}, Transition{5, '0'}, Transition{4, '0'}, Transition{6, '1'}, Transition{0, 0} },
            vector<Transition>{ Transition{0, '0'}, Transition{3, '1'}, Transition{4, '0'}, Transition{6, '1'} }
        },
        SingleWaveformTestPair{
            vector<Transition>{ Transition{}, Transition{0, '0'}, Transition{5, 'x'}, Transition{5, 'y'}, Transition{0, 0} },
            vector<Transition>{ Transition{0, '0'}, Transition{5, 'y'} }
        },
        SingleWaveformTestPair{
            vector<Transition>{ Transition{}, Transition{0, '0'}, Transition{5, '1'}, Transition{4, '0'}, Transition{6, '1'}, Transition{3, '0'} },
            vector<Transition>{ Transition{0, '0'} }
        },
        SingleWaveformTestPair{
            vector<Transition>{ Transition{}, Transition{10, '0'}, Transition{11, '1'}, Transition{0, '0'}, Transition{5, '1'} },
            vector<Transition>{ Transition{0, '0'}, Transition{5, '1'} }
        },
        SingleWaveformTestPair{
            vector<Transition>{ Transition{}, Transition{0, 0} },
            vector<Transition>{}
        }
    )
);

struct BatchWaveformTestPair{
    vector<vector<Transition>> batch_waveform;
    vector<Transition> expected;
    unsigned int capacity = 32;
};

class BatchWaveformCollisionTestFixture: public ::testing::TestWithParam<BatchWaveformTestPair>{};

TEST_P(BatchWaveformCollisionTestFixture, SimpleCases) {
    const auto& test_pair = GetParam();
    const auto& batch_waveform = test_pair.batch_waveform;
    const auto num_stimuli = batch_waveform.size();
    const auto capacity =  test_pair.capacity;

    auto w = new Transition[capacity * num_stimuli];
    auto stimuli_lengths = new unsigned int[num_stimuli];
    for (int i_stimuli = 0; i_stimuli < num_stimuli; i_stimuli++) {
        unsigned int stimuli_length = batch_waveform[i_stimuli].size();
        stimuli_lengths[i_stimuli] = stimuli_length;
        for (int idx = 0; idx < stimuli_length; idx++) {
            w[capacity * i_stimuli + idx] = batch_waveform[i_stimuli][idx];
        }
    }
    unsigned int length = 0;
    resolve_collisions_for_batch_waveform(w, capacity, stimuli_lengths, &length, num_stimuli);

    EXPECT_EQ(length, test_pair.expected.size() - 1);

    unsigned int err_num = 0;
    for (int i = 0; i < length; i++) {
        if (w[i] != test_pair.expected[i]) err_num++;
    }
    EXPECT_EQ(err_num, 0);

    delete[] w;
}

INSTANTIATE_TEST_CASE_P(
    GateTests,
    BatchWaveformCollisionTestFixture,
    ::testing::Values(
        BatchWaveformTestPair{
            vector<vector<Transition>>{
                vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{10, '0'} },
                vector<Transition>{ Transition{3, '0'}, Transition{5, '1'} }
            },
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{3, '0'}, Transition{5, '1'}, Transition{0, 0} }
        },
        BatchWaveformTestPair{
            vector<vector<Transition>>{
                vector<Transition>{ Transition{5, '0'}, Transition{7, '1'}, Transition{10, '0'} },
                vector<Transition>{ Transition{3, '0'}, Transition{20, '1'} }
            },
            vector<Transition>{ Transition{3, '0'}, Transition{20, '1'}, Transition{0, 0} }
        },
        BatchWaveformTestPair{
            vector<vector<Transition>>{
                vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{10, '0'} },
                vector<Transition>{ Transition{3, '0'}, Transition{5, '1'} },
                vector<Transition>{ Transition{4, '0'}, Transition{20, '1'} }
            },
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{3, '0'}, Transition{20, '1'}, Transition{0, 0} }
        },
        BatchWaveformTestPair{
            vector<vector<Transition>>{
                vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{10, '0'} },
                vector<Transition>{ Transition{3, '1'}, Transition{5, '0'} },
                vector<Transition>{ }
            },
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{5, '0'}, Transition{0, 0} }
        }
    )
);