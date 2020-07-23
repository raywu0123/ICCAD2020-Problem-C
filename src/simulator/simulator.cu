#include <stack>
#include <cassert>

#include "simulator/simulator.h"
#include "simulator/collision_utils.h"
#include "include/progress_bar.h"

using namespace std;


__device__ __host__ void resolve_collisions_for_batch_stimuli(
    Transition** data,
    const unsigned int* lengths,
    const unsigned int num_inputs, const unsigned int num_outputs
) {
//    TODO parallelize
    unsigned int stimuli_lengths[N_STIMULI_PARALLEL];
    for (int i_output = 0; i_output < num_outputs; i_output++) {
        for(int i_stimuli = 0; i_stimuli < N_STIMULI_PARALLEL; i_stimuli++) {
            stimuli_lengths[i_stimuli] = lengths[num_outputs * i_stimuli + i_output];
            assert(stimuli_lengths[i_stimuli] <= INITIAL_CAPACITY);
        }
        resolve_collisions_for_batch_waveform(data[num_inputs + i_output], stimuli_lengths, N_STIMULI_PARALLEL);
    }
}


__device__ __host__ bool OOB(unsigned int index, Transition** const data, unsigned int i) {
    return index >= N_STIMULI_PARALLEL * INITIAL_CAPACITY or data[i][index].value == 0;
}

__device__ __host__ void prepare_stimuli_head(
    Timestamp s_timestamps[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    char s_values[MAX_NUM_MODULE_ARGS][N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    Transition** data,
    unsigned int write_stimuli_index,
    const unsigned int num_wires, unsigned int** progress_updates
) {
    s_timestamps[write_stimuli_index][0] = data[0][*progress_updates[0]].timestamp;
    for (int i = 0; i < num_wires; ++i) {
        s_values[i][write_stimuli_index][0] = data[i][*progress_updates[i]].value;
    }
}

__device__ __host__ void slice_waveforms(
    Timestamp s_timestamps[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    DelayInfo s_delay_infos[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    char s_values[MAX_NUM_MODULE_ARGS][N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    Transition** data,
    const unsigned int num_wires, unsigned int** progress_updates
) {
    memset(s_timestamps, 0, sizeof(Timestamp) * N_STIMULI_PARALLEL * INITIAL_CAPACITY);
    memset(s_delay_infos, 0, sizeof(DelayInfo) * N_STIMULI_PARALLEL * INITIAL_CAPACITY);
    memset(s_values, 0, sizeof(char) * MAX_NUM_MODULE_ARGS * N_STIMULI_PARALLEL * INITIAL_CAPACITY);
    for(int i = 0; i < num_wires; ++i) *progress_updates[i] = 0;

    unsigned int num_finished = 0;
    unsigned int write_stimuli_index = 0, write_transition_index = 1;

    prepare_stimuli_head(s_timestamps, s_values, data, write_stimuli_index, num_wires, progress_updates);
    for (int i = 0; i < num_wires; ++i) if (data[i][1].value == 0 ) num_finished++;

    while (num_finished < num_wires) {
        // find min timestamp
        Timestamp min_t = LONG_LONG_MAX;
        for (int i = 0; i < num_wires; ++i) {
            const auto& index = *progress_updates[i];
            if (OOB(index + 1, data, i)) continue;
            const auto& t = data[i][index + 1].timestamp;
            if (t < min_t) min_t = t;
        }
        assert(min_t != LONG_LONG_MAX);

        // find advancing wires
        unsigned int advancing[MAX_NUM_MODULE_ARGS], num_advancing = 0;
        for(int i = 0; i < num_wires; ++i) {
            auto& index = *progress_updates[i];
            if (OOB(index + 1, data, i)) continue;
            if (data[i][index + 1].timestamp != min_t) continue;
            advancing[num_advancing] = i; num_advancing++;
        }

        // decide where to write
        if (write_transition_index + num_advancing - 1 >= INITIAL_CAPACITY) {
            write_transition_index = 1; write_stimuli_index++;
            if (write_stimuli_index >= N_STIMULI_PARALLEL) break;
            prepare_stimuli_head(s_timestamps, s_values, data, write_stimuli_index, num_wires, progress_updates);
        }
        if (write_stimuli_index >= N_STIMULI_PARALLEL) break;

        // advance indices
        for (int i = 0; i < num_advancing; ++i) {
            auto& index = *progress_updates[advancing[i]];
            index++;
            if (OOB(index + 1, data, advancing[i])) num_finished++;
        }
        for (int i = 0; i < num_advancing; ++i) {
            s_timestamps[write_stimuli_index][write_transition_index + i] = min_t;

            const auto& advancing_arg = advancing[i];
            s_delay_infos[write_stimuli_index][write_transition_index + i].arg = advancing_arg;
            s_delay_infos[write_stimuli_index][write_transition_index + i].edge_type = get_edge_type(
                data[advancing_arg][*progress_updates[advancing_arg] - 1].value,
                data[advancing_arg][*progress_updates[advancing_arg]].value
            );
            for (int j = 0; j < num_wires; ++j) {
                const auto& transition = data[j][*progress_updates[j]];
                s_values[j][write_stimuli_index][write_transition_index + i] = transition.value;
            }
        }
        write_transition_index += num_advancing;
    }
}

__device__ Timestamp sliced_input_timestamps[N_CELL_PARALLEL][N_STIMULI_PARALLEL][INITIAL_CAPACITY];
__device__ DelayInfo sliced_input_delay_infos[N_CELL_PARALLEL][N_STIMULI_PARALLEL][INITIAL_CAPACITY];
__device__ char sliced_input_values[N_CELL_PARALLEL][MAX_NUM_MODULE_ARGS][N_STIMULI_PARALLEL][INITIAL_CAPACITY];

__device__ void simulate_module(
    const ModuleSpec* const module_spec,
    const SDFSpec* const sdf_spec,
    Transition** const data,
    unsigned int** progress_updates,
    Timestamp s_input_timestamps[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    DelayInfo s_input_delay_infos[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    char s_input_values[MAX_NUM_MODULE_ARGS][N_STIMULI_PARALLEL][INITIAL_CAPACITY]
) {
    if (threadIdx.x == 0) {
        slice_waveforms(
            s_input_timestamps, s_input_delay_infos, s_input_values,
            data, module_spec->num_module_input, progress_updates
        );
    }
    __syncthreads();

//    pack sliced data
    unsigned stimuli_idx = threadIdx.x;
    Transition sliced_data[MAX_NUM_MODULE_ARGS][INITIAL_CAPACITY];
    for (unsigned int k = 0; k < INITIAL_CAPACITY; ++k) {
        const auto& delay_info = s_input_delay_infos[stimuli_idx][k];
        const auto& timestamp = s_input_timestamps[stimuli_idx][k];
        if (s_input_values[0][stimuli_idx][k] == 0) break;
        for (unsigned int i = 0; i < module_spec->num_module_input; ++i) {
            auto& entry = sliced_data[i][k];
            entry.timestamp = timestamp; entry.delay_info = delay_info; entry.value = s_input_values[i][stimuli_idx][k];
        }
    }
    Transition* data_ptrs_for_each_stimuli[MAX_NUM_MODULE_ARGS] = { nullptr };
    for (unsigned int i = 0; i < module_spec->num_module_input; ++i) {
        data_ptrs_for_each_stimuli[i] = sliced_data[i];
    }
    for (unsigned int i = module_spec->num_module_input; i < module_spec->num_module_args; ++i) {
        data_ptrs_for_each_stimuli[i] = data[i] + stimuli_idx * INITIAL_CAPACITY;
    }

    unsigned int offset = 0;
    for (int i = 0; i < module_spec->schedule_size; i++) {
        const unsigned int num_gate_args = module_spec->num_inputs[i] + module_spec->num_outputs[i];
        assert(num_gate_args <= MAX_NUM_GATE_ARGS);
        Transition* data_schedule_for_gate[MAX_NUM_GATE_ARGS] = { nullptr };
        for (int j = 0; j < num_gate_args; ++j) {
            const auto& arg = module_spec->gate_specs[offset + j];
            data_schedule_for_gate[j] = data_ptrs_for_each_stimuli[arg];
        }
        module_spec->gate_schedule[i](
            data_schedule_for_gate,
            module_spec->tables[i], module_spec->table_row_num[i],
            module_spec->num_inputs[i], module_spec->num_outputs[i]
        );
        offset += num_gate_args;
    }
    assert(module_spec->num_module_output <= MAX_NUM_MODULE_OUTPUT);
    __shared__ unsigned int lengths[N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT];
    compute_delay(
        data_ptrs_for_each_stimuli,
        module_spec->num_module_output, module_spec->num_module_input,
        sdf_spec, lengths + stimuli_idx * module_spec->num_module_output
    );

    __syncthreads();
    if (threadIdx.x == 0) {
        resolve_collisions_for_batch_stimuli(
            data, lengths,
            module_spec->num_module_input, module_spec->num_module_output
        );
    }
}

__global__ void simulate_batch(BatchResource batch_resource) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& offset = batch_resource.data_schedule_offsets[blockIdx.x];
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& sdf_spec = batch_resource.sdf_specs[blockIdx.x];
        auto* module_data = &batch_resource.data_schedule[offset];
        auto* progress_updates = &batch_resource.progress_updates[offset];
        simulate_module(
            module_spec, sdf_spec, module_data, progress_updates,
            sliced_input_timestamps[blockIdx.x],
            sliced_input_delay_infos[blockIdx.x],
            sliced_input_values[blockIdx.x]
        );
    }
}

void Simulator::run() {
    cout << "| Status: Running Simulation... " << endl;

    size_t new_heap_size = N_CELL_PARALLEL * N_STIMULI_PARALLEL * MAX_NUM_GATE_ARGS * sizeof(Transition) * INITIAL_CAPACITY * 4;
    cudaErrorCheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size));
    cout << "| Adjusted heap size to be " << new_heap_size  << " bytes" << endl;

    unsigned int num_layers = circuit.cell_schedule.size();
    cout << "| Total " << num_layers << " layers" << endl;

    ProgressBar progress_bar(num_layers);
    for (unsigned int i_layer = 0; i_layer < num_layers; i_layer++) {
        const auto& schedule_layer = circuit.cell_schedule[i_layer];
        stack<Cell*, std::vector<Cell*>> job_queue(schedule_layer);
        int session_id = 0;

        while (not job_queue.empty()) {
            unordered_set<Cell*> processing_cells;
            ResourceBuffer resource_buffer;
            for (int i = 0; i < N_CELL_PARALLEL; i++) {
                if (job_queue.empty()) break;
                auto* cell = job_queue.top(); job_queue.pop(); processing_cells.insert(cell);
                cell->prepare_resource(session_id, resource_buffer);
            }
            BatchResource batch_data{}; batch_data.init(resource_buffer);
            simulate_batch<<<N_CELL_PARALLEL, N_STIMULI_PARALLEL>>>(batch_data);
            cudaDeviceSynchronize();

            for (auto* cell : processing_cells) {
                cell->gather_results();
                if (not cell->finished()) job_queue.push(cell);
            }
            batch_data.free();
            session_id++;
        }
        progress_bar.Progressed(i_layer + 1);
    }
    cout << endl;
}
