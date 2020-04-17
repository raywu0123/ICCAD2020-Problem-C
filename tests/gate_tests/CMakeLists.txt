include(CTest)
include_directories(${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

add_executable(gate_tests builtin_gate_tests.cu)

target_link_libraries(gate_tests gtest gtest_main)
target_link_libraries(gate_tests mylib)

add_test(
        NAME gate_tests
        COMMAND gate_tests
)