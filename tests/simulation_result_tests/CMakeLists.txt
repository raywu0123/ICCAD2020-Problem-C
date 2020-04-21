include(CTest)
include_directories(${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

add_executable(simulation_result_tests util_tests.cu)

target_link_libraries(simulation_result_tests gtest gtest_main)
target_link_libraries(simulation_result_tests mylib)

add_test(
        NAME simulation_result_tests
        COMMAND simulation_result_tests
)