#include "gtest/gtest.h"
#include <vector>

#include "circuit_model/cell.h"

using namespace std;

TEST(circuit_test, trivial) {
    int a = 1, b = 2;
    int c = a + b;
    EXPECT_EQ(c, 3);
}

struct SingleWaveformTestPair {
    vector<vector<Timestamp>> transition_timestamps;
    vector<vector<unsigned int>> expected_schedule_indices;
};

class CellBuildBucketIndexScheduleTestFixture : public ::testing::TestWithParam<SingleWaveformTestPair> {
};

TEST_P(CellBuildBucketIndexScheduleTestFixture, cases) {
    vector<IndexedWire> wires;
    auto& test_pair = GetParam();

    for (const auto& wire_transitions : test_pair.transition_timestamps) {
        auto* w = new Wire();
        for (const auto& t : wire_transitions)
        w->bucket.transitions.emplace_back(t, 0);
        wires.emplace_back(w);
    }

    Cell::build_bucket_index_schedule(wires, 3);

    int num_error = 0;
    const auto& expected_schedule_indices = test_pair.expected_schedule_indices;
    ASSERT_EQ(test_pair.expected_schedule_indices.size(), wires.size());
    for (int i = 0; i < wires.size(); i++) {
        ASSERT_EQ(wires[i].bucket_index_schedule.size(), expected_schedule_indices[i].size());
        for (int j = 0; j < expected_schedule_indices[i].size(); j++) {
            if (wires[i].bucket_index_schedule[j] != expected_schedule_indices[i][j]) num_error++;
        }
    }
    ASSERT_EQ(num_error, 0);
    for (const auto& indexed_wire : wires) {
        delete indexed_wire.wire;
    }
}

INSTANTIATE_TEST_SUITE_P(
    CircuitTests,
    CellBuildBucketIndexScheduleTestFixture,
    ::testing::Values(
        SingleWaveformTestPair{
            vector<vector<Timestamp>> {
                {0, 3,     6, 9, 12,  15, 18, 21,  24, 27, 30},
                {0, 2, 5,  10,                     31}
            },
            vector<vector<unsigned int>> {
                {0, 2, 5, 8, 11},
                {0, 3, 4, 4, 5}
            }
        },
        SingleWaveformTestPair{
            vector<vector<Timestamp>> {
                {0, 3, 5},
                {0, 1}
            },
            vector<vector<unsigned int>> { {0, 3}, {0, 2} }
        },
        SingleWaveformTestPair{
            vector<vector<Timestamp>> {
                {0, 3, 5, 10},
                {0, 100}
            },
            vector<vector<unsigned int>> { {0, 3, 4}, {0, 1, 2} }
        },
        SingleWaveformTestPair{
            vector<vector<Timestamp>> {
                {0, 100, 200, 400},
                {0, 100, 300, 400, 500, 600, 700, 800}
            },
            vector<vector<unsigned int>> { {0, 3, 4, 4}, {0, 2, 5, 8} }
        }
    )
);