#include "simulator/simulator.h"
#include "include/progress_bar.h"

using namespace std;

__global__ void simulate_batch(const BatchResource& batch_resource) {
    printf("$ %d\n", batch_resource.num_modules);
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
            unsigned int cell_idx = i_batch_gate * N_GATE_PARALLEL;
            for (; cell_idx < (i_batch_gate + 1) * N_GATE_PARALLEL and cell_idx < layer_size; cell_idx++) {
                resource_buffer.push_back(schedule_layer[cell_idx]->prepare_resource());
            }

            simulate_batch<<<N_GATE_PARALLEL, N_STIMULI_PARALLEL>>> (
                get_batch_data()
            ); // perform edge checking in the kernel
            cudaDeviceSynchronize();

            for (unsigned int free_cell_idx = i_batch_gate * N_GATE_PARALLEL; free_cell_idx < cell_idx; free_cell_idx++) {
                schedule_layer[free_cell_idx]->free_resource();
//              accumulators will collect results at this stage
            }
        }
    }
}

void Simulator::set_input(vector<unsigned long>& stimuli_indices) const {
    for (int i_wire = 0; i_wire < input_waveforms.num_buckets; i_wire++) {
        auto stimuli_start_index = stimuli_indices[i_wire];
        const auto& bucket = input_waveforms.buckets[i_wire];
        Wire* wire = circuit.input_wires[i_wire];
        unsigned int size = min(
            stimuli_start_index + N_STIMULI_PARALLEL * wire->capacity, bucket.transitions.size()
        ) - stimuli_start_index;
        wire->set_input(bucket.transitions, stimuli_start_index, size);
        stimuli_indices[i_wire] += size;
    }
}

BatchResource Simulator::get_batch_data() {
    BatchResource batch_resource{};
    unsigned int num_modules = resource_buffer.size();
    batch_resource.num_modules = num_modules;

    cudaMalloc((void**) &batch_resource.module_specs, sizeof(ModuleSpec*) * num_modules);
    cudaMalloc((void**) &batch_resource.data_schedule, sizeof(Transition*) * num_modules);
    cudaMalloc((void**) &batch_resource.data_schedule_offsets, sizeof(unsigned int) * num_modules);
    cudaMalloc((void**) &batch_resource.capacities, sizeof(unsigned int*) * num_modules);

    cudaMemcpy(batch_resource.module_specs, &resource_buffer.module_specs[0], sizeof(ModuleSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(batch_resource.data_schedule, &resource_buffer.data_schedule[0], sizeof(Transition*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(batch_resource.data_schedule_offsets, &resource_buffer.data_schedule_offsets[0], sizeof(unsigned int) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(batch_resource.capacities, &resource_buffer.capacities[0], sizeof(unsigned int*) * num_modules, cudaMemcpyHostToDevice);
    resource_buffer.clear();

    return batch_resource;
}
