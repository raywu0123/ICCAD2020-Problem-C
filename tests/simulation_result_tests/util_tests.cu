#include "gtest/gtest.h"
#include <vector>

#include <simulator/data_structures.h>
#include "simulation_result.h"


using namespace std;

struct SingleWaveformTestPair {
    vector<Timestamp> timestamps;
    vector<pair<Timestamp, int>> expected_ans;
};

class GroupTimestampsTestFixture: public ::testing::TestWithParam<SingleWaveformTestPair>
{};

TEST_P(GroupTimestampsTestFixture, Cases) {
    const auto test_case = GetParam();
    const auto& timestamps = test_case.timestamps;
    const auto& expected_ans = test_case.expected_ans;

    vector<pair<Timestamp, int>> timestamp_groups;
    VCDResult::group_timestamps(timestamps, timestamp_groups);

    EXPECT_EQ(timestamp_groups.size(), expected_ans.size());
    for (int i = 0; i < timestamp_groups.size(); i++) {
        EXPECT_EQ(timestamp_groups[i], expected_ans[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GroupTimestampsTest,
    GroupTimestampsTestFixture,
    ::testing::Values(
            SingleWaveformTestPair{
            vector<Timestamp>{0, 0, 1, 2, 3, 3, 3, 3, 4},
            vector<pair<Timestamp, int>> {
                pair<Timestamp, int>{0, 2},
                pair<Timestamp, int>{1, 1},
                pair<Timestamp, int>{2, 1},
                pair<Timestamp, int>{3, 4},
                pair<Timestamp, int>{4, 1}
            }
        }
    )
);