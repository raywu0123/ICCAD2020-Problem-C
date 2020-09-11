#include <cassert>
#include <chrono>
#include <unistd.h>

#include "simulator/simulator.h"
#include "simulator/collision_utils.h"
#include "include/progress_bar.h"

using namespace std;


__device__ __host__ void resolve_collisions_for_batch_stimuli(
    Transition* const all_data, unsigned int* const all_size, Data* data,
    const CAPACITY_TYPE* lengths, const CAPACITY_TYPE& capacity,
    const NUM_ARG_TYPE& num_outputs
) {
//    TODO parallelize
    CAPACITY_TYPE stimuli_lengths[N_STIMULI_PARALLEL];
    for (NUM_ARG_TYPE i_output = 0; i_output < num_outputs; i_output++) {
        if (data[i_output].is_dummy) continue;

        for(int i_stimuli = 0; i_stimuli < N_STIMULI_PARALLEL; i_stimuli++) {
            stimuli_lengths[i_stimuli] = lengths[num_outputs * i_stimuli + i_output];
            assert(stimuli_lengths[i_stimuli] <= capacity);
        }
        resolve_collisions_for_batch_waveform(
            all_data + data[i_output].transition_offset,
            stimuli_lengths, capacity, all_size + data[i_output].size_offset,
            N_STIMULI_PARALLEL
        );
    }
}


__device__ __host__ bool OOB(unsigned int index, InputData* const data, unsigned int i) {
    return index >= data[i].size;
}

__device__ __host__ void prepare_stimuli_head(
    Timestamp* s_timestamps, Values* s_values,
    const Transition* const all_input_data, InputData* data,
    const NUM_ARG_TYPE& num_wires, const CAPACITY_TYPE* progress_updates
) {
    bool is_head = true;
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) is_head &= (all_input_data[data[i].offset + progress_updates[i]].timestamp == 0);

    s_timestamps[0] = is_head ? -1 : all_input_data[data[0].offset + progress_updates[0]].timestamp;
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
        s_values[i] = all_input_data[data[i].offset + progress_updates[i]].value;
    }
}

__device__ __host__ void slice_waveforms(
    Timestamp* s_timestamps, DelayInfo* s_delay_infos, Values* s_values,
    const Transition* const all_input_data, InputData* data, const CAPACITY_TYPE& capacity,
    const NUM_ARG_TYPE& num_wires,
    bool* overflow_ptr
) {
    CAPACITY_TYPE progress[MAX_NUM_MODULE_OUTPUT] = {0};

    NUM_ARG_TYPE num_finished = 0;
    unsigned int write_stimuli_index = 0, write_transition_index = 1;
    auto C = capacity * num_wires;

    prepare_stimuli_head(
        s_timestamps + write_stimuli_index * capacity,
        s_values + write_stimuli_index * C,
        all_input_data, data, num_wires, progress
    );
    for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) if (data[i].size <= 1) num_finished++;

    while (num_finished < num_wires) {
        // find min timestamp and find advancing wires
        Timestamp min_t = LONG_LONG_MAX;
        NUM_ARG_TYPE advancing[MAX_NUM_MODULE_ARGS], num_advancing = 0;

        for (NUM_ARG_TYPE i = 0; i < num_wires; ++i) {
            const auto& index = progress[i];
            if (OOB(index + 1, data, i)) continue;
            const auto& t = all_input_data[data[i].offset + index + 1].timestamp;
            if (t <= min_t) {
                if (t < min_t) {
                    min_t = t;
                    num_advancing = 0;
                }
                advancing[num_advancing] = i;
                num_advancing++;
            }
        }
        assert(min_t != LONG_LONG_MAX);

        // decide where to write
        if (write_transition_index + num_advancing - 1 >= capacity) {
            write_transition_index = 1; write_stimuli_index++;
            if (write_stimuli_index >= N_STIMULI_PARALLEL) break;
            prepare_stimuli_head(
                s_timestamps + write_stimuli_index * capacity,
                s_values + write_stimuli_index * C,
                all_input_data, data, num_wires, progress
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
                all_input_data[data[advancing_arg].offset + progress[advancing_arg] - 1].value,
                all_input_data[data[advancing_arg].offset + progress[advancing_arg]].value
            );
            for (NUM_ARG_TYPE j = 0; j < num_wires; ++j) {
                const auto& transition = all_input_data[data[j].offset + progress[j]];
                s_values[
                    write_stimuli_index * C
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

__device__ void slice_module(
    const ModuleSpec* const module_spec,
    const Transition* const all_input_data, InputData* const input_data,
    const CAPACITY_TYPE& capacity,
    bool* overflow_ptr,
    Timestamp* s_input_timestamps, DelayInfo* s_input_delay_infos, Values* s_input_values
) {
    slice_waveforms(
        s_input_timestamps, s_input_delay_infos, s_input_values,
        all_input_data, input_data, capacity,
        module_spec->num_input, overflow_ptr
    );
}
__global__ void slice_kernel(
    BatchResource batch_resource, Transition* input_data,
    Timestamp* s_timestamps, DelayInfo* s_delay_infos, Values* s_values
) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& overflow_ptr = batch_resource.overflows[blockIdx.x];
        const auto& module_input_data = &batch_resource.input_data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& capacity = batch_resource.capacities[blockIdx.x];
        const auto& s_timestamp_offset = batch_resource.s_timestamp_offsets[blockIdx.x];
        const auto& s_delay_info_offset = batch_resource.s_delay_info_offsets[blockIdx.x];
        const auto& s_values_offset = batch_resource.s_value_offsets[blockIdx.x];
        slice_module(
            module_spec,
            input_data, module_input_data,
            capacity, overflow_ptr,
            s_timestamps + s_timestamp_offset, s_delay_infos + s_delay_info_offset, s_values + s_values_offset
        );
    }
}

__device__ void stepping_module(
    const ModuleSpec* const module_spec,
    Transition* const all_output_data, Data* const output_data,
    const CAPACITY_TYPE& capacity,
    Timestamp* s_input_timestamps, Values* s_input_values
) {
    assert(module_spec->num_output <= MAX_NUM_MODULE_OUTPUT);
    Transition* output_data_ptrs_for_stimuli[MAX_NUM_MODULE_OUTPUT] = { nullptr };
    const unsigned int& stimuli_idx = threadIdx.x;
    for (NUM_ARG_TYPE i = 0; i < module_spec->num_output; ++i) {
        if (output_data[i].is_dummy) continue;
        output_data_ptrs_for_stimuli[i] = all_output_data + output_data[i].transition_offset + stimuli_idx * capacity;
    }

    auto offset = stimuli_idx * static_cast<unsigned int>(capacity);
    stepping_algorithm(
        s_input_timestamps + offset,
        s_input_values + offset * static_cast<unsigned int>(module_spec->num_input),
        output_data_ptrs_for_stimuli,
        module_spec,
        capacity
    );
}
__global__ void stepping_kernel(
    BatchResource batch_resource, Transition* output_data,
    Timestamp* s_timestamps, Values* s_values
) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& module_output_data = &batch_resource.output_data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& capacity = batch_resource.capacities[blockIdx.x];
        const auto& s_timestamp_offset = batch_resource.s_timestamp_offsets[blockIdx.x];
        const auto& s_values_offset = batch_resource.s_value_offsets[blockIdx.x];
        stepping_module(
            module_spec,
            output_data, module_output_data,
            capacity,
            s_timestamps + s_timestamp_offset, s_values + s_values_offset
        );
    }
}

__device__ void compute_delay_module(
    const ModuleSpec* const module_spec,
    const SDFPath* const sdf_paths, const unsigned int& sdf_num_rows,
    Transition* const all_output_data, Data* const output_data,
    const CAPACITY_TYPE& capacity,
    DelayInfo* s_input_delay_infos, CAPACITY_TYPE* lengths
) {
    assert(module_spec->num_output <= MAX_NUM_MODULE_OUTPUT);

    Transition* output_data_ptrs_for_stimuli[MAX_NUM_MODULE_OUTPUT] = { nullptr };
    const unsigned int& stimuli_idx = threadIdx.x;
    for (NUM_ARG_TYPE i = 0; i < module_spec->num_output; ++i) {
        if (output_data[i].is_dummy) continue;
        output_data_ptrs_for_stimuli[i] = all_output_data + output_data[i].transition_offset + stimuli_idx * capacity;
    }

    DelayInfo* delay_info_for_stimuli = s_input_delay_infos + stimuli_idx * capacity;
    compute_delay(
        output_data_ptrs_for_stimuli, capacity, delay_info_for_stimuli,
        module_spec->num_output, module_spec->num_input,
        sdf_paths, sdf_num_rows,
        lengths + stimuli_idx * module_spec->num_output
    );
}
__global__ void compute_delay_kernel(
    BatchResource batch_resource, SDFPath* sdf, Transition* output_data,
    DelayInfo* s_delay_infos, CAPACITY_TYPE* s_lengths
) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& sdf_offset = batch_resource.sdf_offsets[blockIdx.x];
        const auto& s_delay_info_offset = batch_resource.s_delay_info_offsets[blockIdx.x];
        const auto& s_lengths_offset = batch_resource.s_length_offsets[blockIdx.x];
        const auto& sdf_num_rows = batch_resource.sdf_num_rows[blockIdx.x];
        const auto& module_output_data = &batch_resource.output_data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& capacity = batch_resource.capacities[blockIdx.x];
        compute_delay_module(
            module_spec,
            sdf + sdf_offset, sdf_num_rows,
            output_data, module_output_data,
            capacity,
            s_delay_infos + s_delay_info_offset,
            s_lengths + s_lengths_offset
        );
    }
}

__device__ void resolve_collision_module(
    const ModuleSpec* const module_spec,
    Transition* const all_output_data, Data* const output_data, unsigned int* const all_size,
    const CAPACITY_TYPE& capacity, CAPACITY_TYPE* lengths
) {
    resolve_collisions_for_batch_stimuli(
        all_output_data, all_size, output_data,
        lengths, capacity, module_spec->num_output
    );
}

__global__ void resolve_collision_kernel(
    BatchResource batch_resource, Transition* output_data, unsigned int* output_size,
    CAPACITY_TYPE* s_lengths
) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& module_output_data = &batch_resource.output_data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& s_length_offset = batch_resource.s_length_offsets[blockIdx.x];
        const auto& capacity = batch_resource.capacities[blockIdx.x];
        resolve_collision_module(
            module_spec,
            output_data, module_output_data, output_size,
            capacity,
            s_lengths + s_length_offset
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
    vector<CellProcessor> cell_processors; cell_processors.resize(N_STREAM);
    for (unsigned int i_layer = 0; i_layer < num_layers; i_layer++) {
        const auto& schedule_layer = circuit.cell_schedule[i_layer];
        const auto& split_cells = split_schedule_layer(schedule_layer, N_STREAM);

        ResourceCollector<SDFPath, Cell> sdf_collector(schedule_layer.size());
        ResourceCollector<Transition, Wire> input_data_collector(schedule_layer.size() * MAX_NUM_MODULE_ARGS);

        for (int i = 0; i < N_STREAM; ++i)
            cell_processors[i].layer_init(split_cells[i], sdf_collector, input_data_collector);

        auto* device_sdf = sdf_collector.get();
        auto* device_input_data = input_data_collector.get();
        for (auto& processor : cell_processors) processor.set_ptrs(device_sdf, device_input_data);
        cudaDeviceSynchronize();

        bool all_finished = false;
        while (not all_finished) {
            all_finished = true;
            for (auto& processor : cell_processors) {
                all_finished &= processor.run();
            }
        }

        sdf_collector.free(); input_data_collector.free();
        progress_bar.Progressed(i_layer + 1);
    }

    cout << endl;
}

vector<vector<Cell*>> Simulator::split_schedule_layer(const vector<Cell*>& layer, unsigned int num_split) {
    vector<vector<Cell*>> splits; splits.resize(num_split);
    int split_size = ceil(double(layer.size()) / double(num_split));
    for (int i = 0; i < num_split; ++i) {
        splits[i].reserve(split_size);
        for (int j = 0; j < split_size; ++j) {
            if (i * split_size + j >= layer.size()) break;
            splits[i].push_back(layer[i * split_size + j]);
        }
    }
    return splits;
}

CellProcessor::CellProcessor() {
    cudaStreamCreate(&stream);
    batch_data.init();
}

CellProcessor::~CellProcessor() {
    cudaStreamDestroy(stream);
    batch_data.free();
    output_data_collector.free(); output_size_collector.free();
    s_timestamp_collector.free(); s_delay_info_collector.free(); s_values_collector.free();
    overflow_collector.free();
}

void CellProcessor::layer_init(
    std::vector<Cell*> cells,
    ResourceCollector<SDFPath, Cell>& sdf_collector, ResourceCollector<Transition, Wire>& input_data_collector
) {
    session_id = 0;
    overflow_collector.reset(); overflow_collector.reserve(cells.size());

    job_queue = stack<Cell*, std::vector<Cell*>>(cells);
    for (auto* cell : cells) cell->init(sdf_collector, input_data_collector, overflow_collector);
}

void CellProcessor::set_ptrs(SDFPath *sdf, Transition *input_data) {
    device_sdf = sdf; device_input_data = input_data;
}

bool CellProcessor::run() {
    if (has_unfinished) return false;
    else if (!has_unfinished and job_queue.empty()) return true;

    processing_cells.clear();
    s_timestamp_collector.reset(); s_delay_info_collector.reset(); s_values_collector.reset(); s_length_collector.reset();
    output_data_collector.reset(); output_size_collector.reset();
    auto* device_overflow = overflow_collector.get_device(stream);
    for (int i = 0; i < N_CELL_PARALLEL; i++) {
        if (job_queue.empty()) break;
        auto* cell = job_queue.top(); processing_cells.insert(cell);
        cell->prepare_resource(
                session_id, resource_buffer, device_overflow,
                output_data_collector, output_size_collector,
                s_timestamp_collector, s_delay_info_collector, s_values_collector, s_length_collector
        );
        if (cell->finished()) job_queue.pop();
    }
    batch_data.set(resource_buffer, stream); resource_buffer.clear();

    auto* device_output_data = output_data_collector.get_device(stream);
    auto* device_s_timestamps = s_timestamp_collector.get_device(stream);
    auto* device_s_delay_infos = s_delay_info_collector.get_device(stream);
    auto* device_s_values = s_values_collector.get_device(stream);
    auto* device_s_lengths = s_length_collector.get_device(stream);
    auto* device_sizes = output_size_collector.get_device(stream);
    slice_kernel<<<N_CELL_PARALLEL, 1, 0, stream>>>(
            batch_data, device_input_data,
            device_s_timestamps, device_s_delay_infos, device_s_values
    );
    stepping_kernel<<<N_CELL_PARALLEL, N_STIMULI_PARALLEL, 0, stream>>>(
            batch_data, device_output_data,
            device_s_timestamps, device_s_values
    );
    compute_delay_kernel<<<N_CELL_PARALLEL, N_STIMULI_PARALLEL, 0, stream>>>(
            batch_data, device_sdf, device_output_data,
            device_s_delay_infos, device_s_lengths
    );
    resolve_collision_kernel<<<N_CELL_PARALLEL, 1, 0, stream>>>(
            batch_data, device_output_data, device_sizes,
            device_s_lengths
    );
    host_output_data = output_data_collector.get_host(stream);
    host_sizes = output_size_collector.get_host(stream);
    host_overflows = overflow_collector.get_host(stream);

    cudaStreamAddCallback(stream, CellProcessor::post_process, (void*) this, 0);
    has_unfinished = true;
    return false;
}

CUDART_CB void CellProcessor::post_process(cudaStream_t stream, cudaError_t status, void* userData) {
    auto* processor = static_cast<CellProcessor*>(userData);

    unordered_set<Cell*> non_overflow_cells, finished_cells;
    for (auto* cell : processor->processing_cells) {
        bool finished = cell->finished();
        bool overflow = cell->handle_overflow(processor->host_overflows);
        if (not overflow) {
            non_overflow_cells.insert(cell);
            if (finished) finished_cells.insert(cell);
        } else if (finished) processor->job_queue.push(cell);
    }

    for (auto* cell : non_overflow_cells) cell->gather_results(processor->host_output_data, processor->host_sizes);
    for (auto* cell : finished_cells) cell->free();
    processor->session_id++;
    processor->has_unfinished = false;
}
