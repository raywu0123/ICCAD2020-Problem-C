set(HEADER_FILES
        input_waveforms.h
        simulator/simulator.h
        simulation_result.h
        circuit_model/circuit.h
        file_readers/intermediate_file_reader.h
        file_readers/vcd_reader.h
        specs/timing_spec.h
        specs/sdf_cell.h
        data_containers.h simulator/builtin_gates.h)
set(SOURCE_FILES
        input_waveforms.cu
        simulator/simulator.cu
        simulation_result.cu
        circuit_model/circuit.cu
        file_readers/intermediate_file_reader.cu
        file_readers/vcd_reader.cu
        specs/timing_spec.cu
        specs/sdf_cell.cu
        data_containers.cu simulator/data_structures.cu simulator/data_structures.h simulator/module_registry.cu simulator/module_registry.h)
add_library(mylib STATIC ${HEADER_FILES} ${SOURCE_FILES})
