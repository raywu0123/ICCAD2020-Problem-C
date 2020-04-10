#include "simulator/simulator.h"

void Simulator::run() {

}


__global__ void simulate_batch(
        const ModuleFnPtr* modules_fn_ptrs, ModuleSpec* module_specs, int module_num,
        char** const data_schedules, int** const timestamp_schedules, const int* data_schedule_offsets,
        const int* gate_data_schedules_offsets, const int* gate_data_schedule_offset_offsets,
        const int* capacities, const int* capacities_offsets,
        const int* gate_capacity_offsets, const int* gate_capacity_offset_offsets,
        const int n_stimuli_parallel
) {
    // data_schedules, first star: capacity, stimuli. second star: wires, gates.
    int module_idx = blockIdx.x;
    if (module_idx < module_num) {
        modules_fn_ptrs[module_idx](
            module_specs[module_idx],
            data_schedules + data_schedule_offsets[module_idx],
            timestamp_schedules + data_schedule_offsets[module_idx],
            gate_data_schedules_offsets + gate_data_schedule_offset_offsets[module_idx],
            capacities + capacities_offsets[module_idx],
            gate_capacity_offsets + gate_data_schedule_offset_offsets[module_idx],
            n_stimuli_parallel
        );
    }
};
