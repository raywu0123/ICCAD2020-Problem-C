include(CTest)
include_directories(${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

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

add_executable(delay_handling_tests delay_handling_tests.cu)
target_link_libraries(delay_handling_tests gtest gtest_main)
target_link_libraries(delay_handling_tests mylib)
add_test(
        NAME delay_handling_tests
        COMMAND delay_handling_tests
)
