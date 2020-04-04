#include "gtest/gtest.h"


TEST(circuit_test, trivial) {
    int a = 1, b = 2;
    int c = a + b;
    EXPECT_EQ(c, 3);
}