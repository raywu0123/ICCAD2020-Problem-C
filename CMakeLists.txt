cmake_minimum_required(VERSION 3.10)
project(ICCAD2020 CUDA)

add_executable(GPUSimulator GPUSimulator.cu)

set_target_properties(
    GPUSimulator
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
