set(HEADER_FILES
        vcd_reader.h
        simulator/simulator.h
        simulator/data_structures.h
        simulator/module_registry.h
        simulator/memory_manager.h
        simulator/collision_utils.h
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
        simulator/memory_manager.cu
        simulator/delay_utils.cu
        simulator/collision_utils.cu
        simulation_result.cu
        circuit_model/circuit.cu
        circuit_model/cell.cu
        circuit_model/wire.cu
        circuit_model/wire_wrappers.cu
        include/progress_bar.cu
        utils.cu
        )

set(CMAKE_CUDA_FLAGS "-g -G")
add_library(mylib STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(mylib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
