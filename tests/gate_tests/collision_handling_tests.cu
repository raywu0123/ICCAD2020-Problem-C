#include "gtest/gtest.h"
#include <vector>

#include "simulator/simulator.h"

using namespace std;


struct BatchWaveformTestPair{
    vector<vector<Transition>> batch_waveform;
    vector<Transition> expected;
    unsigned int capacity = 16;
};

class BatchWaveformCollisionTestFixture: public ::testing::TestWithParam<BatchWaveformTestPair>{};

TEST_P(BatchWaveformCollisionTestFixture, SimpleCases) {
    const auto& test_pair = GetParam();
    const auto& capacity = test_pair.capacity;
    const auto& batch_waveform = test_pair.batch_waveform;
    const auto num_stimuli = batch_waveform.size();

    auto w = new Transition[capacity * num_stimuli];
    auto stimuli_lengths = new unsigned int[num_stimuli];
    for (int i_stimuli = 0; i_stimuli < num_stimuli; i_stimuli++) {
        unsigned int stimuli_length = batch_waveform[i_stimuli].size();
        stimuli_lengths[i_stimuli] = stimuli_length;
        for (int idx = 0; idx < stimuli_length; idx++) {
            w[capacity * i_stimuli + idx] = batch_waveform[i_stimuli][idx];
        }
    }
    unsigned int output_length = 0;
    resolve_collisions_for_batch_waveform(w, stimuli_lengths, capacity, &output_length, num_stimuli);

    EXPECT_EQ(output_length, test_pair.expected.size());
    unsigned int err_num = 0;
    for (int i = 0; i < test_pair.expected.size() - 1; i++) {
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
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{3, '0'}, Transition{5, '1'} }
        },
        BatchWaveformTestPair{
            vector<vector<Transition>>{
                vector<Transition>{ Transition{5, '0'}, Transition{7, '1'}, Transition{10, '0'} },
                vector<Transition>{ Transition{3, '0'}, Transition{20, '1'} }
            },
            vector<Transition>{ Transition{3, '0'}, Transition{20, '1'} }
        },
        BatchWaveformTestPair{
            vector<vector<Transition>>{
                vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{10, '0'} },
                vector<Transition>{ Transition{3, '0'}, Transition{5, '1'} },
                vector<Transition>{ Transition{4, '0'}, Transition{20, '1'} }
            },
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{3, '0'}, Transition{20, '1'} }
        },
        BatchWaveformTestPair{
            vector<vector<Transition>>{
                vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{10, '0'} },
                vector<Transition>{ Transition{3, '1'}, Transition{5, '0'} },
                vector<Transition>{ }
            },
            vector<Transition>{ Transition{0, '0'}, Transition{1, '1'}, Transition{5, '0'} }
        }
    )
);
