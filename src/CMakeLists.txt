set(HEADER_FILES
        input_waveforms.h
        simulator/simulator.h
        simulator/data_structures.h
        simulator/module_registry.h
        simulator/memory_manager.h
        simulator/builtin_gates.h
        simulation_result.h
        circuit_model/circuit.h
        circuit_model/cell.h
        circuit_model/wire.h
        accumulators.h
        include/progress_bar.h
        utils.h)
set(SOURCE_FILES
        input_waveforms.cu
        simulator/simulator.cu
        simulator/data_structures.cu
        simulator/module_registry.cu
        simulator/memory_manager.cu
        simulation_result.cu
        circuit_model/circuit.cu
        circuit_model/cell.cu
        circuit_model/wire.cu
        accumulators.cu
        include/progress_bar.cu
        utils.cu simulator/builtin_gates.cu)
add_library(mylib STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(mylib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
