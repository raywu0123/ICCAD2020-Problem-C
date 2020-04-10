#ifndef ICCAD2020_BUILTIN_GATES_H
#define ICCAD2020_BUILTIN_GATES_H


// Gates parallel compute on n_stimuli_parallel
__device__ void and_gate_fn(
        char** const data, int** const timestamps, const int num_inputs, const int num_outputs,
        const char* table, const int* capacities, const int n_stimuli_parallel
) {
    int stimuli_index = threadIdx.x;
    printf("stimuli index: %d\n", stimuli_index);

    if (stimuli_index < n_stimuli_parallel) {

    }
}

__device__ GateFnPtr and_gate_fn_ptr = and_gate_fn;


__device__ void PrimitiveGate(
        char** const data, int** const timestamps, const int num_inputs, const int num_outputs,
        const char* table, const int* capacities, const int n_stimuli_parallel
) {

};


// compute single module for multiple stimuli
__device__ void Module(
        const ModuleSpec& module_spec,
        char** const data_schedule,
        int** const timestamp_schedule,
        const int* data_schedule_offsets,
        const int* capacities,
        const int* capacities_offsets,
        const int n_stimuli_parallel
) {
    for(int i_schedule = 0; i_schedule < module_spec.schedule_size; i_schedule++) {
        // compute one gate
        GateFnPtr gate = module_spec.gate_schedule[i_schedule];
        char* table = module_spec.tables[i_schedule];
        gate(
            data_schedule + data_schedule_offsets[i_schedule],
            timestamp_schedule + data_schedule_offsets[i_schedule],
            module_spec.num_inputs[i_schedule], module_spec.num_outputs[i_schedule],
            table,
            capacities + capacities_offsets[i_schedule],
            n_stimuli_parallel
        );
    }
}

#endif //ICCAD2020_BUILTIN_GATES_H
