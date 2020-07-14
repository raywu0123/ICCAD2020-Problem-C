#include "gtest/gtest.h"
#include <vector>

#include "circuit_model/cell.h"

using namespace std;

struct TestPair {
    vector<vector<Transition>> buckets;
    vector<vector<Transition>> ans_scheduled_buckets;
    vector<unsigned int> ans_starting_indices;
};

class ScheduledBucketTestFixture: public ::testing::TestWithParam<TestPair>{};

TEST_P(ScheduledBucketTestFixture, SimpleCases) {
    auto test_pair = GetParam();
    auto buckets = test_pair.buckets;
    auto num_wires = test_pair.buckets.size();

    vector<Wire> inner_wires; inner_wires.resize(num_wires);
    for (int i = 0; i < num_wires; i++) inner_wires[i].bucket.transitions = buckets[i];

    vector<ScheduledWire*> scheduled_wires;
    for (auto& w : inner_wires) scheduled_wires.push_back(new ScheduledWire(&w));

    vector<unsigned int> starting_indices;
    Cell::build_scheduled_buckets(scheduled_wires, starting_indices);

    const auto& ans_scheduled_buckets = test_pair.ans_scheduled_buckets;
    for (int i = 0; i < num_wires; i++) {
        ASSERT_EQ(scheduled_wires[i]->scheduled_bucket, ans_scheduled_buckets[i]);
    }
    const auto& ans_starting_indices = test_pair.ans_starting_indices;
    ASSERT_EQ(starting_indices, ans_starting_indices);
}

INSTANTIATE_TEST_SUITE_P(
    GateTests,
    ScheduledBucketTestFixture,
    ::testing::Values(
        TestPair{
            vector<vector<Transition>> {{
                Transition{0, 'x'}, Transition{1, '1'}, Transition{2, '0'}
            }},
            vector<vector<Transition>> {{
                Transition{0, 'x'}, Transition{1, '1', DelayInfo{0, '+'}}, Transition{2, '0', DelayInfo{0, '-'}}
            }},
            vector<unsigned int> {1, 2, 3}
        },
        TestPair{
            vector<vector<Transition>> {
                { Transition{0, 'x'}, Transition{1, '1'}, Transition{2, '0'} },
                { Transition{0, 'x'}, Transition{1, '0'}, Transition{3, 'z'} }
            },
            vector<vector<Transition>> {
                { Transition{0, 'x'}, Transition{1, '1', DelayInfo{0, '+'}}, Transition{1, '1', DelayInfo{1, '-'}},
                  Transition{2, '0', DelayInfo{0, '-'}}, Transition{3, '0', DelayInfo{1, '+'}} },
                { Transition{0, 'x'}, Transition{1, '0', DelayInfo{0, '+'}}, Transition{1, '0', DelayInfo{1, '-'}},
                  Transition{2, '0', DelayInfo{0, '-'}}, Transition{3, 'z', DelayInfo{1, '+'}} }
            },
            vector<unsigned int> {1, 3, 4, 5}
        }
    )
);
