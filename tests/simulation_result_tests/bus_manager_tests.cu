#include "gtest/gtest.h"
#include <string>

#include "circuit_model/circuit.h"

using namespace std;

struct TestPair {
    unsigned int index;
    string expected_output;
};


class IndexToIdentifierTestFixture: public ::testing::TestWithParam<TestPair>
{};

TEST_P(IndexToIdentifierTestFixture, TestIndexToIdentifier) {
    auto test_pair = GetParam();
    unsigned int index = test_pair.index;
    const string expected_ans = test_pair.expected_output;

    const auto& out = BusManager::index_to_identifier(index);

    EXPECT_EQ(out, expected_ans);
}

INSTANTIATE_TEST_SUITE_P(
    IndexToIdentifierTest,
    IndexToIdentifierTestFixture,
    ::testing::Values(
        TestPair{0, "0"},
        TestPair{15, "f"},
        TestPair{16, "10"}
    )
);