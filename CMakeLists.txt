cmake_minimum_required(VERSION 3.10)
project(ICCAD2020 CUDA)

set(HEADER_FILES
        src/file_readers/intermediate_file_reader.h
        src/graph.h
        src/specs/timing_spec.h
        src/file_readers/vcd_reader.h
        src/input_waveforms.h
        src/simulator.h
        src/simulation_result.h
        src/specs/standard_cell.h
        src/specs/gv_cell.h
        src/data_containers.h)
set(SOURCE_FILES
        src/file_readers/intermediate_file_reader.cu
        src/graph.cu
        src/specs/timing_spec.cu
        src/file_readers/vcd_reader.cu
        src/input_waveforms.cu
        src/simulator.cu
        src/simulation_result.cu
        src/specs/standard_cell.cu
        src/specs/gv_cell.cu
        src/data_containers.cu)
add_library(mylib STATIC ${HEADER_FILES} ${SOURCE_FILES})

add_executable(GPUSimulator GPUSimulator.cu src/file_readers/intermediate_file_reader.h src/file_readers/intermediate_file_reader.cu)

set_target_properties(
    GPUSimulator
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(GPUSimulator mylib)

