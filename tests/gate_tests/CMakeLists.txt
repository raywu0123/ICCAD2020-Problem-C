include(CTest)
include_directories(${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

add_executable(builtin_gate_tests builtin_gate_tests.cu)
target_link_libraries(builtin_gate_tests gtest gtest_main)
target_link_libraries(builtin_gate_tests mylib)
add_test(
        NAME builtin_gate_tests
        COMMAND builtin_gate_tests
)

add_executable(simulator_tests simulator_tests.cu)
target_link_libraries(simulator_tests gtest gtest_main)
target_link_libraries(simulator_tests mylib)
add_test(
        NAME simulator_tests
        COMMAND simulator_tests
)

add_executable(collision_handling_tests collision_handling_tests.cu)
target_link_libraries(collision_handling_tests gtest gtest_main)
target_link_libraries(collision_handling_tests mylib)
add_test(
        NAME collision_handling_tests
        COMMAND collision_handling_tests
)

