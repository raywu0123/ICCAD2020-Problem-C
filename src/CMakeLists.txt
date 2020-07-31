set(HEADER_FILES
        vcd_reader.h
        simulator/simulator.h
        simulator/data_structures.h
        simulator/module_registry.h
        simulator/collision_utils.h
        simulator/device_functions.h
        simulator/job.h
        simulation_result.h
        circuit_model/circuit.h
        circuit_model/cell.h
        circuit_model/wire.h
        include/progress_bar.h
        utils.h
        )
set(SOURCE_FILES
        vcd_reader.cu
        simulator/simulator.cu
        simulator/data_structures.cu
        simulator/module_registry.cu
        simulator/delay_utils.cu
        simulator/collision_utils.cu
        simulator/device_functions.cu
        simulator/job.cu
        simulation_result.cu
        circuit_model/circuit.cu
        circuit_model/cell.cu
        circuit_model/wire.cu
        include/progress_bar.cu
        utils.cu
        )

set(CMAKE_CUDA_FLAGS "-g -G")
add_library(mylib STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(mylib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
