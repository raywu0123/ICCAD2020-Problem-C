include(CTest)
include_directories(${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

add_executable(circuit_tests circuit_tests.cu)

target_link_libraries(circuit_tests gtest gtest_main)
target_link_libraries(circuit_tests mylib)

add_test(
        NAME circuit_tests
        COMMAND circuit_tests
)