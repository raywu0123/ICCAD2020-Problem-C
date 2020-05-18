#include "gtest/gtest.h"
#include <vector>

#include "simulator/data_structures.h"
#include "circuit_model/cell.h"
#include "simulator/simulator.h"

using namespace std;

struct SingleWaveformTestPair{
    vector<vector<Transition>> data_schedule;
    unsigned int num_inputs;
    vector<SDFPath> sdf_paths;
    vector<vector<Timestamp>> expected_output_timestamps;
};

class ComputeDelayTestFixture: public ::testing::TestWithParam<SingleWaveformTestPair>
{
protected:
};


TEST_P(ComputeDelayTestFixture, SimpleCases) {
    const auto& test_pair = GetParam();
    const auto& vector_data_schedule = test_pair.data_schedule;
    const auto& sdf_paths = test_pair.sdf_paths;
    const auto& expected_output_timestamps = test_pair.expected_output_timestamps;

    const unsigned int num_module_inputs = test_pair.num_inputs;
    const unsigned int num_module_outputs = vector_data_schedule.size() - num_module_inputs;

    auto** data_schedule = new Transition*[vector_data_schedule.size()];
    auto* data_schedule_indices = new unsigned int[vector_data_schedule.size()];
    auto* capacities = new unsigned int[vector_data_schedule.size()];
    for (int i = 0; i < vector_data_schedule.size(); i++) {
        data_schedule[i] = const_cast<Transition *>(vector_data_schedule[i].data());
        data_schedule_indices[i] = i;
        capacities[i] = vector_data_schedule[i].size();
    }

    SDFSpec sdf_spec{};
    sdf_spec.num_rows = sdf_paths.size();
    sdf_spec.input_index = new unsigned int[sdf_spec.num_rows];
    sdf_spec.output_index = new unsigned int[sdf_spec.num_rows];
    sdf_spec.edge_type = new char[sdf_spec.num_rows];
    sdf_spec.rising_delay = new int[sdf_spec.num_rows];
    sdf_spec.falling_delay = new int[sdf_spec.num_rows];
    for (int i = 0; i < sdf_spec.num_rows; i++) {
        sdf_spec.input_index[i] = sdf_paths[i].in;
        sdf_spec.output_index[i] = sdf_paths[i].out;
        sdf_spec.edge_type[i] = sdf_paths[i].edge_type;
        sdf_spec.rising_delay[i] = sdf_paths[i].rising_delay;
        sdf_spec.falling_delay[i] = sdf_paths[i].falling_delay;
    }

    compute_delay(
        data_schedule,
        vector_data_schedule.size(),
        capacities,
        data_schedule_indices,
        num_module_inputs, num_module_outputs,
        &sdf_spec
    );

    int num_error = 0;
    for (int i_output = 0; i_output < num_module_outputs; i_output++) {
        for (int i = 1; i < expected_output_timestamps[i_output].size(); i++) {
            if (expected_output_timestamps[i_output][i] != vector_data_schedule[num_module_inputs + i_output][i].timestamp)
                num_error++;
        }
    }

    EXPECT_EQ(num_error, 0);

    delete[] data_schedule;
    delete[] data_schedule_indices;
    delete[] sdf_spec.input_index;
    delete[] sdf_spec.edge_type;
    delete[] sdf_spec.rising_delay;
    delete[] sdf_spec.falling_delay;
}

INSTANTIATE_TEST_SUITE_P(
    GateTests,
    ComputeDelayTestFixture,
    ::testing::Values(
            SingleWaveformTestPair{
            vector<vector<Transition>>{
                { Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'z'} },
                { Transition{0, '0'}, Transition{2, '1'}, Transition{3, '0'}, Transition{4, '1'} },
                { Transition{0, '0'}, Transition{1, '0'}, Transition{2, 'x'}, Transition{2, 'x'} }
            }, 2,
            vector<SDFPath> {
                SDFPath{0, 2, 'x', 10, 10},
                SDFPath{1, 2, 'x', 5, 5},
            },
            vector<vector<Timestamp>>{ {0, 11, 12, 7} }
        },
            SingleWaveformTestPair{
            vector<vector<Transition>>{
                { Transition{0, '0'}, Transition{1, '1'}, Transition{2, 'x'}, Transition{3, 'z'} },
                { Transition{0, '0'}, Transition{2, '1'}, Transition{3, '0'}, Transition{4, '1'} },
                { Transition{0, '0'}, Transition{1, '0'}, Transition{2, 'x'}, Transition{2, 'x'} }
            }, 2,
            vector<SDFPath> {
                SDFPath{0, 2, '+', 9, 9},
                SDFPath{0, 2, '-', 10, 10},
                SDFPath{1, 2, '+', 6, 6},
                SDFPath{1, 2, '-', 5, 5},
            },
            vector<vector<Timestamp>>{ {0, 10, 12, 8} }
        },
            SingleWaveformTestPair{
            vector<vector<Transition>>{
                { Transition{0, '0'}, Transition{1, 'x'}, Transition{2, 'z'}, Transition{3, 'x'} },
                { Transition{0, '0'}, Transition{2, 'z'}, Transition{3, 'x'}, Transition{4, '1'} },
                { Transition{0, '0'}, Transition{1, '0'}, Transition{2, 'z'}, Transition{2, 'x'} }
            }, 2,
            vector<SDFPath> {
                SDFPath{0, 2, 'x', 9, 9},
                SDFPath{1, 2, 'x', 5, 5},
            },
            vector<vector<Timestamp>>{ {0, 10, 11, 7} }
        },
            SingleWaveformTestPair{
            vector<vector<Transition>>{
                { Transition{0, '0'}, Transition{1, 'x'}, Transition{2, 'z'}, Transition{3, 'x'} },
                { Transition{0, '0'}, Transition{2, 'z'}, Transition{3, 'x'}, Transition{4, '1'} },
                { Transition{0, '0'}, Transition{1, '0'}, Transition{2, 'z'}, Transition{2, 'x'} },
                { Transition{0, '0'}, Transition{1, '0'}, Transition{2, 'z'}, Transition{2, 'x'} }
            }, 2,
            vector<SDFPath> {
                SDFPath{0, 2, 'x', 10, 10},
                SDFPath{0, 3, 'x', 8, 8},
                SDFPath{1, 2, 'x', 5, 5},
                SDFPath{1, 3, 'x', 6, 6},
            },
            vector<vector<Timestamp>>{ {0, 11, 12, 7}, {0, 9, 10, 8} }
        }
    )
);
