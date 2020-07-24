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
    Timestamp s_timestamps[INITIAL_CAPACITY],
    char s_values[INITIAL_CAPACITY][MAX_NUM_MODULE_ARGS],
    Transition** data,
    const unsigned int num_wires, unsigned int** progress_updates
) {
    s_timestamps[0] = data[0][*progress_updates[0]].timestamp;
    for (int i = 0; i < num_wires; ++i) {
        s_values[0][i] = data[i][*progress_updates[i]].value;
    }
}

__device__ __host__ void slice_waveforms(
    Timestamp s_timestamps[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    DelayInfo s_delay_infos[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    char s_values[N_STIMULI_PARALLEL][INITIAL_CAPACITY][MAX_NUM_MODULE_ARGS],
    Transition** data,
    const unsigned int num_wires, unsigned int** progress_updates
) {
    memset(s_timestamps, 0, sizeof(Timestamp) * N_STIMULI_PARALLEL * INITIAL_CAPACITY);
    memset(s_delay_infos, 0, sizeof(DelayInfo) * N_STIMULI_PARALLEL * INITIAL_CAPACITY);
    memset(s_values, 0, sizeof(char) * MAX_NUM_MODULE_ARGS * N_STIMULI_PARALLEL * INITIAL_CAPACITY);
    for(int i = 0; i < num_wires; ++i) *progress_updates[i] = 0;

    unsigned int num_finished = 0;
    unsigned int write_stimuli_index = 0, write_transition_index = 1;

    prepare_stimuli_head(
        s_timestamps[write_stimuli_index], s_values[write_stimuli_index],
        data, num_wires, progress_updates
    );
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
            prepare_stimuli_head(
                s_timestamps[write_stimuli_index], s_values[write_stimuli_index],
                data, num_wires, progress_updates
            );
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
                s_values[write_stimuli_index][write_transition_index + i][j] = transition.value;
            }
        }
        write_transition_index += num_advancing;
    }
}

__host__ __device__ unsigned int get_table_row_index(
    const char s_input_values[MAX_NUM_MODULE_ARGS], unsigned int num_input
) {
    unsigned int row_index = 0;
    for (unsigned int i_input = 0; i_input < num_input; ++i_input) {
        unsigned int v;
        switch (s_input_values[i_input]) {
            case '0':
                v = 0;
                break;
            case '1':
                v = 1;
                break;
            case 'x':
                v = 2;
                break;
            case 'z':
                v = 3;
                break;
        }
        row_index = (row_index << 2) + v;
    }
    return row_index;
}
__host__ __device__ void stepping_algorithm(
    const Timestamp s_input_timestamps[INITIAL_CAPACITY],
    const char s_input_values[INITIAL_CAPACITY][MAX_NUM_MODULE_ARGS],
    Transition** output_data,
    const ModuleSpec* module_spec
) {
    for (unsigned int i = 0; i < INITIAL_CAPACITY; i++) {
        if (s_input_values[i][0] == 0) break;
        auto row_index = get_table_row_index(s_input_values[i], module_spec->num_input);
        for (unsigned int j = 0; j < module_spec->num_output; ++j) {
            output_data[j][i].value = module_spec->table[row_index * module_spec->num_output + j];
            output_data[j][i].timestamp = s_input_timestamps[i];
        }
    }
}

__device__ Timestamp sliced_input_timestamps[N_CELL_PARALLEL][N_STIMULI_PARALLEL][INITIAL_CAPACITY];
__device__ DelayInfo sliced_input_delay_infos[N_CELL_PARALLEL][N_STIMULI_PARALLEL][INITIAL_CAPACITY];
__device__ char sliced_input_values[N_CELL_PARALLEL][N_STIMULI_PARALLEL][INITIAL_CAPACITY][MAX_NUM_MODULE_ARGS];

__device__ void simulate_module(
    const ModuleSpec* const module_spec,
    const SDFSpec* const sdf_spec,
    Transition** const data,
    unsigned int** progress_updates,
    Timestamp s_input_timestamps[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    DelayInfo s_input_delay_infos[N_STIMULI_PARALLEL][INITIAL_CAPACITY],
    char s_input_values[N_STIMULI_PARALLEL][INITIAL_CAPACITY][MAX_NUM_MODULE_ARGS]
) {
    if (threadIdx.x == 0) {
        slice_waveforms(
            s_input_timestamps, s_input_delay_infos, s_input_values,
            data, module_spec->num_input, progress_updates
        );
    }
    __syncthreads();

    assert(module_spec->num_output <= MAX_NUM_MODULE_OUTPUT);
    Transition* output_data_ptrs_for_stimuli[MAX_NUM_MODULE_OUTPUT] = { nullptr };
    unsigned stimuli_idx = threadIdx.x;
    for (unsigned int i = 0; i < module_spec->num_output; ++i) {
        output_data_ptrs_for_stimuli[i] = data[module_spec->num_input + i] + stimuli_idx * INITIAL_CAPACITY;
    }

    stepping_algorithm(
        s_input_timestamps[stimuli_idx],
        s_input_values[stimuli_idx],
        output_data_ptrs_for_stimuli,
        module_spec
    );

    assert(module_spec->num_output <= MAX_NUM_MODULE_OUTPUT);
    __shared__ unsigned int lengths[N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT];
    DelayInfo* delay_info_for_stimuli = s_input_delay_infos[stimuli_idx];
    compute_delay(
        output_data_ptrs_for_stimuli, delay_info_for_stimuli,
        module_spec->num_output, module_spec->num_input,
        sdf_spec, lengths + stimuli_idx * module_spec->num_output
    );

    __syncthreads();
    if (threadIdx.x == 0) {
        resolve_collisions_for_batch_stimuli(
            data, lengths,
            module_spec->num_input, module_spec->num_output
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
            cudaDeviceSynchronize(); // since async memcpy
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
