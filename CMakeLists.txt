cmake_minimum_required(VERSION 3.10)
project(ICCAD2020 CUDA)

include_directories(src)
add_subdirectory(src)

add_executable(
        GPUSimulator
        GPUSimulator.cu
)

set_target_properties(
        GPUSimulator
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(GPUSimulator mylib)

enable_testing()
add_subdirectory(tests)

