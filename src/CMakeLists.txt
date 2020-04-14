set(HEADER_FILES
        input_waveforms.h
        simulator/simulator.h
        simulator/data_structures.h
        simulator/module_registry.h
        simulator/builtin_gates.h
        simulation_result.h
        circuit_model/circuit.h
        utils.h
        include/progress_bar.h)
set(SOURCE_FILES
        input_waveforms.cu
        simulator/simulator.cu
        simulator/data_structures.cu
        simulator/module_registry.cu
        simulation_result.cu
        circuit_model/circuit.cu
        include/progress_bar.cu accumulators.cu accumulators.h simulator/memory_manager.cu simulator/memory_manager.h circuit_model/cell.cu circuit_model/cell.h circuit_model/wire.cu circuit_model/wire.h)
add_library(mylib STATIC ${HEADER_FILES} ${SOURCE_FILES})
