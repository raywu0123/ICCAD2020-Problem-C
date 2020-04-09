#include "simulator/simulator.h"

void Simulator::run() {

}

__global__ void simulate_batch(
        Module** modules, int module_num,
        char*** data_schedules, const int* data_schedule_offsets
) {
    int module_idx = blockIdx.x;
    int stimuli_idx = threadIdx.x;
    if (    module_idx < module_num
            and data_schedule_offsets[module_idx] + stimuli_idx < data_schedule_offsets[module_idx + 1]
            ) {
        modules[module_idx]->compute(data_schedules[data_schedule_offsets[module_idx] + stimuli_idx]);
    }
};
