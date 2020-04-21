#include "simulator/simulator.h"
#include "include/progress_bar.h"

using namespace std;

__device__ void simulate_gate_on_multiple_stimuli(
    GateFnPtr gate_fn_ptr,
    Transition** data,  //(n_stimuli * capacities[i_wire], num_inputs + num_outputs)
    unsigned int* capacities,
    char* table,
    unsigned int table_row_num,
    unsigned int num_inputs, unsigned int num_outputs
) {
    unsigned int stimuli_idx = threadIdx.x;
    auto** stimuli_data = new Transition*[num_inputs + num_outputs]; // (capacities[i], num_inputs + num_outputs)
    for (int i = 0; i < num_inputs + num_outputs; i++) {
        stimuli_data[i] = data[i] + capacities[i] * stimuli_idx;
    }
    gate_fn_ptr(stimuli_data, capacities, table, table_row_num, num_inputs, num_outputs);
    delete[] stimuli_data;
}

__device__ void simulate_module(const ModuleSpec* module_spec, Transition** data_schedule, unsigned int* capacities) {
    unsigned int data_schedule_idx = 0;
    for (int i = 0; i < module_spec->schedule_size; i++) {
        const auto& gate_fn_ptr = module_spec->gate_schedule[i];
        const auto& table = module_spec->tables[i];
        const auto& table_row_num = module_spec->table_row_num[i];
        const auto& num_inputs = module_spec->num_inputs[i];
        const auto& num_outputs = module_spec->num_outputs[i];
        simulate_gate_on_multiple_stimuli(
            gate_fn_ptr,
            data_schedule + data_schedule_idx,
            capacities + data_schedule_idx,
            table,
            table_row_num,
            num_inputs, num_outputs
        );
        data_schedule_idx += num_inputs + num_outputs;
    }
}

__global__ void simulate_batch(BatchResource batch_resource) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& offset = batch_resource.data_schedule_offsets[blockIdx.x];
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        auto module_data_schedule = &batch_resource.data_schedule[offset];
        auto module_capacities = &batch_resource.capacities[offset];
        simulate_module(module_spec, module_data_schedule, module_capacities);
    }
};

void Simulator::run() {
    unsigned int num_batches = (int) ceil(double(input_waveforms.num_stimuli) / double(N_STIMULI_PARALLEL));
    cout << "Total " << num_batches << " batches" << endl;
    ProgressBar bar(num_batches, "Running Simulation");
    for (unsigned int i_batch = 0; i_batch < num_batches; i_batch++) {
        simulate_batch_stimuli(i_batch);
        bar.Progressed(i_batch);
    }
    cout << endl;
}

void Simulator::simulate_batch_stimuli(unsigned int& i_batch) {
    set_input(i_batch);

    for (const auto& schedule_layer : circuit.cell_schedule) {
        int n_batch_gate = ceil(double(schedule_layer.size()) / double(N_GATE_PARALLEL));
        int layer_size = schedule_layer.size();

        for (int i_batch_gate = 0; i_batch_gate < n_batch_gate; i_batch_gate++) {
            unsigned int cell_idx = i_batch_gate * N_GATE_PARALLEL;
            resource_buffer.module_specs.reserve(N_GATE_PARALLEL);
            resource_buffer.data_schedule_offsets.reserve(N_GATE_PARALLEL);
            resource_buffer.data_schedule.reserve(N_GATE_PARALLEL * 3);
            resource_buffer.capacities.reserve(N_GATE_PARALLEL * 3);
            for (; cell_idx < (i_batch_gate + 1) * N_GATE_PARALLEL and cell_idx < layer_size; cell_idx++) {
                schedule_layer[cell_idx]->prepare_resource(resource_buffer);
            }

            const auto& batch_data = get_batch_data();
            simulate_batch<<<N_GATE_PARALLEL, N_STIMULI_PARALLEL>>> (batch_data);
            // perform edge checking in the kernel
            cudaDeviceSynchronize();

            for (unsigned int free_cell_idx = i_batch_gate * N_GATE_PARALLEL; free_cell_idx < cell_idx; free_cell_idx++) {
                schedule_layer[free_cell_idx]->free_resource();
//              accumulators will collect results at this stage
            }
        }
    }
}

void Simulator::set_input(unsigned int i_batch) const {
    for (int i_wire = 0; i_wire < input_waveforms.num_buckets; i_wire++) {
        const auto& bucket = input_waveforms.buckets[i_wire];
        auto& wire_ptr = bucket.wire_ptr;

        for (unsigned int i_stimuli = 0; i_stimuli < N_STIMULI_PARALLEL; i_stimuli++) {
            unsigned int global_i_stimuli = i_batch * N_STIMULI_PARALLEL + i_stimuli;
            if (global_i_stimuli >= input_waveforms.num_stimuli) break;

            wire_ptr->set_input(
                bucket.transitions,
                bucket.stimuli_edge_indices,
                global_i_stimuli
            );
        }
    }
}

BatchResource Simulator::get_batch_data() {
    BatchResource batch_resource{};
    unsigned int num_modules = resource_buffer.size();
    batch_resource.num_modules = num_modules;

    cudaMalloc((void**) &batch_resource.module_specs, sizeof(ModuleSpec*) * num_modules);
    cudaMalloc((void**) &batch_resource.data_schedule, sizeof(Transition*) * resource_buffer.data_schedule.size());
    cudaMalloc((void**) &batch_resource.data_schedule_offsets, sizeof(unsigned int) * num_modules);
    cudaMalloc((void**) &batch_resource.capacities, sizeof(unsigned int) * resource_buffer.capacities.size());

    cudaMemcpy(batch_resource.module_specs, resource_buffer.module_specs.data(), sizeof(ModuleSpec*) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(batch_resource.data_schedule, resource_buffer.data_schedule.data(), sizeof(Transition*) * resource_buffer.data_schedule.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(batch_resource.data_schedule_offsets, resource_buffer.data_schedule_offsets.data(), sizeof(unsigned int) * num_modules, cudaMemcpyHostToDevice);
    cudaMemcpy(batch_resource.capacities, resource_buffer.capacities.data(), sizeof(unsigned int) * resource_buffer.capacities.size(), cudaMemcpyHostToDevice);
    resource_buffer.clear();

    return batch_resource;
}
