include(CTest)
include_directories(${GTEST_SOURCE_DIR}/include ${GTEST_SOURCE_DIR})

add_executable(slicing_tests slicing_tests.cu)
target_link_libraries(slicing_tests gtest gtest_main)
target_link_libraries(slicing_tests mylib)
add_test(
        NAME slicing_tests
        COMMAND slicing_tests
)

add_executable(stepping_tests stepping_tests.cu)
target_link_libraries(stepping_tests gtest gtest_main)
target_link_libraries(stepping_tests mylib)
add_test(
        NAME stepping_tests
        COMMAND stepping_tests
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
