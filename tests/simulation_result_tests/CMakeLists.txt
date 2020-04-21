include(CTest)
include_directories(${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

add_executable(util_tests util_tests.cu)

target_link_libraries(util_tests gtest gtest_main)
target_link_libraries(util_tests mylib)
add_test(
        NAME util_tests
        COMMAND util_tests
)

add_executable(bus_manager_tests bus_manager_tests.cu)
target_link_libraries(bus_manager_tests gtest gtest_main)
target_link_libraries(bus_manager_tests mylib)
add_test(
        NAME bus_manager_tests
        COMMAND bus_manager_tests
)
