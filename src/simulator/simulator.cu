#include <stack>
#include <cassert>

#include "simulator/simulator.h"
#include "simulator/collision_utils.h"
#include "include/progress_bar.h"

using namespace std;


__device__ __host__ void resolve_collisions_for_batch_stimuli(
    Data* data,
    const CAPACITY_TYPE* lengths, const CAPACITY_TYPE& capacity,
    const NUM_ARG_TYPE& num_inputs, const NUM_ARG_TYPE& num_outputs
) {
//    TODO parallelize
    CAPACITY_TYPE stimuli_lengths[N_STIMULI_PARALLEL];
    for (NUM_ARG_TYPE i_output = 0; i_output < num_outputs; i_output++) {
        if (data[num_inputs + i_output].transitions == nullptr) continue;

        for(int i_stimuli = 0; i_stimuli < N_STIMULI_PARALLEL; i_stimuli++) {
            stimuli_lengths[i_stimuli] = lengths[num_outputs * i_stimuli + i_output];
            assert(stimuli_lengths[i_stimuli] <= capacity);
        }
        resolve_collisions_for_batch_waveform(
            data[num_inputs + i_output].transitions,
            stimuli_lengths, capacity, data[num_inputs + i_output].size,
            N_STIMULI_PARALLEL
        );
    }
}


__device__ __host__ bool OOB(unsigned int index, Data* const data, unsigned int i) {
    return index >= N_STIMULI_PARALLEL * INITIAL_CAPACITY or data[i].transitions[index].value == Values::PAD;
}

__device__ __host__ void prepare_stimuli_head(
    Timestamp* s_timestamps, Values* s_values,
    Data* data,
    const NUM_ARG_TYPE& num_wires, const CAPACITY_TYPE* progress_updates
) {
    bool is_head = true;
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) is_head &= (data[i].transitions[progress_updates[i]].timestamp == 0);

    s_timestamps[0] = is_head ? -1 : data[0].transitions[progress_updates[0]].timestamp;
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
        s_values[i] = data[i].transitions[progress_updates[i]].value;
    }
}

__device__ __host__ void slice_waveforms(
    Timestamp* s_timestamps, DelayInfo* s_delay_infos, Values* s_values,
    Data* data, const CAPACITY_TYPE& capacity,
    const NUM_ARG_TYPE& num_wires,
    bool* overflow_ptr
) {
    memset(s_timestamps, 0, sizeof(Timestamp) * N_STIMULI_PARALLEL * capacity);
    memset(s_delay_infos, 0, sizeof(DelayInfo) * N_STIMULI_PARALLEL * capacity);
    memset(s_values, 0, sizeof(Values) * num_wires * N_STIMULI_PARALLEL * capacity);
    CAPACITY_TYPE progress[MAX_NUM_MODULE_OUTPUT] = {0};

    NUM_ARG_TYPE num_finished = 0;
    unsigned int write_stimuli_index = 0, write_transition_index = 1;

    prepare_stimuli_head(
        s_timestamps + write_stimuli_index * capacity,
        s_values + write_stimuli_index * capacity * num_wires,
        data, num_wires, progress
    );
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) if (data[i].transitions[1].value == Values::PAD) num_finished++;

    while (num_finished < num_wires) {
        // find min timestamp
        Timestamp min_t = LONG_LONG_MAX;
        for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
            const auto& index = progress[i];
            if (OOB(index + 1, data, i)) continue;
            const auto& t = data[i].transitions[index + 1].timestamp;
            if (t < min_t) min_t = t;
        }
        assert(min_t != LONG_LONG_MAX);

        // find advancing wires
        NUM_ARG_TYPE advancing[MAX_NUM_MODULE_ARGS], num_advancing = 0;
        for(NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
            auto& index = progress[i];
            if (OOB(index + 1, data, i)) continue;
            if (data[i].transitions[index + 1].timestamp != min_t) continue;
            advancing[num_advancing] = i; num_advancing++;
        }

        // decide where to write
        if (write_transition_index + num_advancing - 1 >= capacity) {
            write_transition_index = 1; write_stimuli_index++;
            if (write_stimuli_index >= N_STIMULI_PARALLEL) break;
            prepare_stimuli_head(
                s_timestamps + write_stimuli_index * capacity,
                s_values + write_stimuli_index * capacity * num_wires,
                data, num_wires, progress
            );
        }
        // advance indices
        for (NUM_ARG_TYPE i = 0; i < num_advancing; ++i) {
            auto& index = progress[advancing[i]];
            index++;
            if (OOB(index + 1, data, advancing[i])) num_finished++;
        }
        for (NUM_ARG_TYPE i = 0; i < num_advancing; ++i) {
            s_timestamps[write_stimuli_index * capacity + write_transition_index + i] = min_t;

            const auto& advancing_arg = advancing[i];
            s_delay_infos[write_stimuli_index * capacity + write_transition_index + i].arg = advancing_arg;
            s_delay_infos[write_stimuli_index * capacity + write_transition_index + i].edge_type = get_edge_type(
                data[advancing_arg].transitions[progress[advancing_arg] - 1].value,
                data[advancing_arg].transitions[progress[advancing_arg]].value
            );
            for (NUM_ARG_TYPE j = 0; j < num_wires; ++j) {
                const auto& transition = data[j].transitions[progress[j]];
                s_values[
                    write_stimuli_index * capacity * num_wires
                    + (write_transition_index + i) * num_wires
                    + j
                ] = transition.value;
            }
        }
        write_transition_index += num_advancing;
    }
    if (write_stimuli_index >= N_STIMULI_PARALLEL) *overflow_ptr = true;
}

__host__ __device__ unsigned int get_table_row_index(const Values* s_input_values, NUM_ARG_TYPE num_input) {
    unsigned int row_index = 0;
    for (NUM_ARG_TYPE i_input = 0; i_input < num_input; ++i_input) {
        row_index = (row_index << 2) + static_cast<unsigned int>(s_input_values[i_input]) - 1;
    }
    return row_index;
}
__host__ __device__ void stepping_algorithm(
    const Timestamp* s_input_timestamps,
    const Values* s_input_values,
    Transition** output_data,
    const ModuleSpec* module_spec,
    const CAPACITY_TYPE& capacity
) {
    for (CAPACITY_TYPE i = 0; i < capacity; i++) {
        if (s_input_values[i * module_spec->num_input] == Values::PAD) break;
        const auto row_index = get_table_row_index(s_input_values + i * module_spec->num_input, module_spec->num_input);
        for (NUM_ARG_TYPE j = 0; j < module_spec->num_output; ++j) {
            if (output_data[j] == nullptr) continue;
            output_data[j][i].value = module_spec->table[row_index * module_spec->num_output + j];
            output_data[j][i].timestamp = s_input_timestamps[i];
        }
    }
}

__device__ void simulate_module(
    const ModuleSpec* const module_spec,
    const SDFSpec* const sdf_spec,
    Data* const data, const CAPACITY_TYPE& capacity,
    bool* overflow_ptr
) {
    __shared__ Timestamp* s_input_timestamps; __shared__ DelayInfo* s_input_delay_infos; __shared__ Values* s_input_values;
    if (threadIdx.x == 0) {
        auto size = N_STIMULI_PARALLEL * static_cast<unsigned int>(capacity);
        s_input_timestamps = new Timestamp[size];
        s_input_delay_infos = new DelayInfo[size];
        s_input_values = new Values[size * static_cast<unsigned int>(module_spec->num_input)];
        slice_waveforms(
            s_input_timestamps, s_input_delay_infos, s_input_values,
            data, capacity,
            module_spec->num_input, overflow_ptr
        );
    }
    __syncthreads();
    assert(module_spec->num_output <= MAX_NUM_MODULE_OUTPUT);
    Transition* output_data_ptrs_for_stimuli[MAX_NUM_MODULE_OUTPUT] = { nullptr };
    const unsigned int& stimuli_idx = threadIdx.x;
    for (NUM_ARG_TYPE i = 0; i < module_spec->num_output; ++i) {
        if (data[module_spec->num_input + i].transitions == nullptr) continue;
        output_data_ptrs_for_stimuli[i] = data[module_spec->num_input + i].transitions + stimuli_idx * capacity;
    }

    auto offset = stimuli_idx * static_cast<unsigned int>(capacity);
    stepping_algorithm(
        s_input_timestamps + offset,
        s_input_values + offset * static_cast<unsigned int>(module_spec->num_input),
        output_data_ptrs_for_stimuli,
        module_spec,
        capacity
    );

    assert(module_spec->num_output <= MAX_NUM_MODULE_OUTPUT);
    __shared__ CAPACITY_TYPE lengths[N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT];
    DelayInfo* delay_info_for_stimuli = s_input_delay_infos + stimuli_idx * capacity;
    compute_delay(
        output_data_ptrs_for_stimuli, capacity, delay_info_for_stimuli,
        module_spec->num_output, module_spec->num_input,
        sdf_spec, lengths + stimuli_idx * module_spec->num_output
    );

    __syncthreads();
    if (threadIdx.x == 0) {
        resolve_collisions_for_batch_stimuli(
            data, lengths, capacity,
            module_spec->num_input, module_spec->num_output
        );
        delete[] s_input_timestamps; delete[] s_input_delay_infos; delete[] s_input_values;
    }
}

__global__ void simulate_batch(BatchResource batch_resource) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& sdf_spec = batch_resource.sdf_specs[blockIdx.x];
        const auto& overflow_ptr = batch_resource.overflows[blockIdx.x];
        const auto& module_data = &batch_resource.data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& capacity = batch_resource.capacities[blockIdx.x];
        simulate_module(
            module_spec, sdf_spec, module_data, capacity, overflow_ptr
        );
    }
}

void Simulator::run() {
    cout << "| Status: Running Simulation... " << endl;

    size_t new_heap_size = N_CELL_PARALLEL * N_STIMULI_PARALLEL * INITIAL_CAPACITY * 8
            * (sizeof(Timestamp) + sizeof(DelayInfo) + sizeof(Values) * MAX_NUM_MODULE_ARGS);
    cudaErrorCheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size));
    cout << "| Adjusted heap size to be " << new_heap_size  << " bytes" << endl;

    unsigned int num_layers = circuit.cell_schedule.size();
    cout << "| Total " << num_layers << " layers" << endl;

    ProgressBar progress_bar(num_layers);
    ResourceBuffer resource_buffer;
    BatchResource batch_data{}; batch_data.init();
    for (unsigned int i_layer = 0; i_layer < num_layers; i_layer++) {
        const auto& schedule_layer = circuit.cell_schedule[i_layer];
        stack<Cell*, std::vector<Cell*>> job_queue(schedule_layer);
        for (auto* cell : schedule_layer) cell->init();
        int session_id = 0;

        while (not job_queue.empty()) {
            unordered_set<Cell*> processing_cells;
            for (int i = 0; i < N_CELL_PARALLEL; i++) {
                if (job_queue.empty()) break;
                auto* cell = job_queue.top(); processing_cells.insert(cell);
                cell->prepare_resource(session_id, resource_buffer);
                if (cell->finished()) job_queue.pop();
            }
            batch_data.set(resource_buffer); resource_buffer.clear();
            simulate_batch<<<N_CELL_PARALLEL, N_STIMULI_PARALLEL>>>(batch_data);
            cudaDeviceSynchronize();

            for (auto* cell : processing_cells) {
                bool finished = cell->finished();
                bool overflow = cell->gather_results();
                if (finished and overflow) job_queue.push(cell);
            }
            session_id++;
        }

        for (auto* cell : schedule_layer) cell->free();
        progress_bar.Progressed(i_layer + 1);
    }
    batch_data.free();
    cout << endl;
}
