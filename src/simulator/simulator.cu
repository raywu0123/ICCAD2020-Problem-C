#include "simulator/simulator.h"
#include "include/progress_bar.h"


__global__ void simulate_batch(
        BatchResource
//        const ModuleFnPtr* modules_fn_ptrs, ModuleSpec* module_specs, int module_num,
//        char** const data_schedules, int** const timestamp_schedules, const int* data_schedule_offsets,
//        const int* gate_data_schedules_offsets, const int* gate_data_schedule_offset_offsets,
//        const int* capacities, const int* capacities_offsets,
//        const int* gate_capacity_offsets, const int* gate_capacity_offset_offsets,
//        const int n_stimuli_parallel
) {
//    // data_schedules, first star: capacity, stimuli. second star: wires, gates.
//    int module_idx = blockIdx.x;
//    if (module_idx < module_num) {
//        modules_fn_ptrs[module_idx](
//                module_specs[module_idx],
//                data_schedules + data_schedule_offsets[module_idx],
//                timestamp_schedules + data_schedule_offsets[module_idx],
//                gate_data_schedules_offsets + gate_data_schedule_offset_offsets[module_idx],
//                capacities + capacities_offsets[module_idx],
//                gate_capacity_offsets + gate_data_schedule_offset_offsets[module_idx],
//                n_stimuli_parallel
//        );
//    }
};

void Simulator::run() {
    vector<unsigned long> stimuli_indices;

    int num_input_wires = input_waveforms.buckets.size();
    stimuli_indices.resize(num_input_wires);

    ProgressBar bar(input_waveforms.max_transition, "Running Simulation");
    bar.SetFrequencyUpdate(10000);

    auto& progress = stimuli_indices[input_waveforms.max_transition_index];
    while (progress < input_waveforms.max_transition ) {
        simulate_batch_stimuli(stimuli_indices);
        bar.Progressed(progress);
    }
    cout << endl;
}


void Simulator::simulate_batch_stimuli(vector<unsigned long>& stimuli_indices) {
    set_input(stimuli_indices);

    for (const auto& schedule_layer : circuit.cell_schedule) {
        int n_batch_gate = ceil(double(schedule_layer.size()) / double(N_GATE_PARALLEL));
        int layer_size = schedule_layer.size();

        for (int i_batch_gate = 0; i_batch_gate < n_batch_gate; i_batch_gate++) {
            int cell_idx = i_batch_gate * N_GATE_PARALLEL;
            for (; cell_idx < (i_batch_gate + 1) * N_GATE_PARALLEL and cell_idx < layer_size; cell_idx++) {
                update_resource(schedule_layer[cell_idx]->prepare_resource());
            }

            simulate_batch<<<N_GATE_PARALLEL, N_STIMULI_PARALLEL>>> (
                get_batch_data()
            ); // perform edge checking in the kernel
            cudaDeviceSynchronize();

            for (int free_cell_idx = i_batch_gate * N_GATE_PARALLEL; free_cell_idx < cell_idx; free_cell_idx++) {
                schedule_layer[free_cell_idx]->free_resource();
            }
        }
    }
}

void Simulator::set_input(vector<unsigned long>& stimuli_indices) const {
    for (int i_wire = 0; i_wire < input_waveforms.num_buckets; i_wire++) {
        auto stimuli_start_index = stimuli_indices[i_wire];
        const auto& bucket = input_waveforms.buckets[i_wire];
        Wire* wire = circuit.input_wires[i_wire];
        unsigned long size = min(stimuli_start_index + N_STIMULI_PARALLEL, bucket.transitions.size()) - stimuli_start_index;
        wire->set_input(bucket.transitions, stimuli_start_index, size);
        stimuli_indices[i_wire] += size;
    }
}