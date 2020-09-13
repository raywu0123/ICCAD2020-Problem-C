#include <cassert>
#include <unistd.h>
#include <thread>

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

__device__ void slice_module(
    const ModuleSpec* const module_spec,
    Transition* const all_input_data, InputData* const input_data,
    const CAPACITY_TYPE& s_capacity,
    bool* s_overflow_ptr,
    SliceInfo* s_slice_infos
) {
    slice_waveforms(
        s_slice_infos,
        all_input_data, input_data, s_capacity - 1,
        module_spec->num_input, s_overflow_ptr
    );
}
__global__ void slice_kernel(
    BatchResource batch_resource, Transition* input_data, SliceInfo* slice_infos, bool* s_overflows
) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& module_input_data = &batch_resource.input_data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& s_capacity = batch_resource.s_capacities[blockIdx.x];
        const auto& s_overflow_offset = batch_resource.s_overflow_offsets[blockIdx.x];
        const auto& s_slice_info_offset = batch_resource.s_slice_info_offsets[blockIdx.x];
        slice_module(
            module_spec,
            input_data, module_input_data,
            s_capacity, s_overflows + s_overflow_offset,
            slice_infos + s_slice_info_offset
        );
    }
}

__host__ __device__ unsigned int get_table_row_index(Transition** input_data, const unsigned int* progress, NUM_ARG_TYPE num_input) {
    unsigned int row_index = 0;
    for (NUM_ARG_TYPE i_input = 0; i_input < num_input; ++i_input) {
        row_index = (row_index << 2) + static_cast<unsigned int>(input_data[i_input][progress[i_input]].value) - 1;
    }
    return row_index;
}

__host__ __device__ void stepping_algorithm(
    Transition** input_data, const unsigned int* sizes,
    DelayInfo* s_delay_infos,
    Transition** output_data,
    const ModuleSpec* module_spec,
    const CAPACITY_TYPE& capacity,
    bool* overflow_ptr,
    bool verbose
) {
    unsigned int progress[MAX_NUM_MODULE_OUTPUT] = {0};
    NUM_ARG_TYPE num_finished = 0;
    const auto &num_input = module_spec->num_input;
    const auto &num_output = module_spec->num_output;

    bool is_head = true;
    for (NUM_ARG_TYPE i = 0; i < num_input; ++i) {
        is_head &= (input_data[i][0].timestamp == 0);
        if (sizes[i] <= 1) num_finished++;
    }

    const auto& row_index = get_table_row_index(input_data, progress, num_input);
    for (NUM_ARG_TYPE o = 0; o < num_output; ++o) {
        if(output_data[o] == nullptr) continue;
        output_data[o][0].value = module_spec->table[row_index * num_output + o];
        output_data[o][0].timestamp = is_head ? -1 : input_data[0][0].timestamp;
    }

    unsigned int write_index = 1;
    while (num_finished < num_input) {
        // find min_t and advancing_args
        Timestamp min_t = LONG_LONG_MAX;
        NUM_ARG_TYPE advancing[MAX_NUM_MODULE_OUTPUT], num_advancing = 0;
        for (NUM_ARG_TYPE i = 0; i < num_input; ++i) {
            const auto& index = progress[i];
            if (index + 1 >= sizes[i]) continue;
            const auto& t = input_data[i][index + 1].timestamp;
            if (t <= min_t) {
                if (t < min_t) {
                    min_t = t;
                    num_advancing = 0;
                }
                advancing[num_advancing] = i;
                num_advancing++;
            }
        }
        if (write_index + num_advancing - 1 >= capacity) break;
        // advance_indices
        for (NUM_ARG_TYPE i = 0; i < num_advancing; ++i) {
            const auto& advancing_arg = advancing[i];
            auto& index = progress[advancing_arg];
            index++;
            if (index + 1 >= sizes[advancing_arg]) num_finished++;
        }

        const auto& rid = get_table_row_index(input_data, progress, num_input);
        for (NUM_ARG_TYPE i = 0; i < num_advancing; ++i) {
            const auto& advancing_arg = advancing[i];
            s_delay_infos[write_index].arg = advancing_arg;
            s_delay_infos[write_index].edge_type = get_edge_type(
                input_data[advancing_arg][progress[advancing_arg] - 1].value,
                input_data[advancing_arg][progress[advancing_arg]].value
            );

            for (NUM_ARG_TYPE o = 0; o < num_output; ++o) {
                if(output_data[o] == nullptr) continue;
                output_data[o][write_index].value = module_spec->table[rid * num_output + o];
                output_data[o][write_index].timestamp = min_t;
            }
            write_index++;
        }
    }

    if (num_finished != num_input) *overflow_ptr = true;
}

__device__ void stepping_module(
    const ModuleSpec* const module_spec,
    Transition* const all_input_data, InputData* const input_data,
    Transition* const all_output_data, Data* const output_data,
    SliceInfo* s_slice_infos, DelayInfo* s_delay_infos,
    const CAPACITY_TYPE& capacity,
    bool* overflow_ptr
) {
    assert(module_spec->num_output <= MAX_NUM_MODULE_OUTPUT);
    Transition* output_data_ptrs_for_stimuli[MAX_NUM_MODULE_OUTPUT] = { nullptr };
    const unsigned int& stimuli_idx = threadIdx.x;

    const auto offset = stimuli_idx * capacity;
    for (NUM_ARG_TYPE i = 0; i < module_spec->num_output; ++i) {
        if (output_data[i].is_dummy) continue;
        output_data_ptrs_for_stimuli[i] = all_output_data + output_data[i].transition_offset + offset;
    }

    if (not (stimuli_idx != 0 and s_slice_infos[(stimuli_idx + 1) * module_spec->num_input].offset == 0)) {
        Transition* input_data_ptrs[MAX_NUM_MODULE_OUTPUT] = {nullptr};
        unsigned int sizes[MAX_NUM_MODULE_OUTPUT] = {0};
        for (int i = 0; i < module_spec->num_input; ++i)  {
            auto stimuli_offset = s_slice_infos[stimuli_idx * module_spec->num_input + i].offset;
            if (stimuli_offset > 0) stimuli_offset--;
            input_data_ptrs[i] = all_input_data + input_data[i].offset + stimuli_offset;
            sizes[i] = s_slice_infos[(stimuli_idx + 1) * module_spec->num_input + i].offset - stimuli_offset;
        }

        stepping_algorithm(
            input_data_ptrs, sizes,
            s_delay_infos + offset,
            output_data_ptrs_for_stimuli,
            module_spec,
            capacity,
            overflow_ptr
        );
    }
}
__global__ void stepping_kernel(
    BatchResource batch_resource,
    Transition* input_data, Transition* output_data,
    SliceInfo* s_slice_infos, DelayInfo* s_delay_infos,
    bool* overflows
) {
    if (blockIdx.x < batch_resource.num_modules) {
        const auto& module_spec = batch_resource.module_specs[blockIdx.x];
        const auto& module_input_data = &batch_resource.input_data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& module_output_data = &batch_resource.output_data_schedule[blockIdx.x * MAX_NUM_MODULE_ARGS];
        const auto& overflow_offset = batch_resource.overflow_offsets[blockIdx.x];
        const auto& s_slice_info_offset = batch_resource.s_slice_info_offsets[blockIdx.x];
        const auto& s_delay_info_offset = batch_resource.s_delay_info_offsets[blockIdx.x];
        const auto& output_capacity = batch_resource.output_capacities[blockIdx.x];
        stepping_module(
            module_spec,
            input_data, module_input_data,
            output_data, module_output_data,
            s_slice_infos + s_slice_info_offset, s_delay_infos + s_delay_info_offset,
            output_capacity, overflows + overflow_offset
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
        const auto& capacity = batch_resource.output_capacities[blockIdx.x];
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
        const auto& capacity = batch_resource.output_capacities[blockIdx.x];
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
        const auto& split_cells = split_vector(schedule_layer, N_STREAM);

        ResourceCollector<SDFPath, Cell> sdf_collector(schedule_layer.size());
        ResourceCollector<Transition, Wire> input_data_collector(schedule_layer.size() * MAX_NUM_MODULE_ARGS);

        vector<thread> init_threads;
        for (int i = 0; i < N_STREAM; ++i) {
            init_threads.emplace_back(
                CellProcessor::layer_init_async,
                std::ref(cell_processors[i]), std::ref(split_cells[i])
            );
        }
        for (auto& thread : init_threads) thread.join();
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

CellProcessor::CellProcessor() {
    cudaStreamCreate(&stream);
    batch_data.init();
}

CellProcessor::~CellProcessor() {
    cudaStreamDestroy(stream);
    batch_data.free();
    output_data_collector.free(); output_size_collector.free();
    s_slice_info_collector.free();
    overflow_collector.free(); s_overflow_collector.free();
}

void CellProcessor::layer_init_async(CellProcessor& processor, const std::vector<Cell*>& cells) {
    processor.session_id = 0;
    processor.overflow_collector.reset(); processor.overflow_collector.reserve(cells.size());
    processor.s_overflow_collector.reset(); processor.s_overflow_collector.reserve(cells.size());
    processor.job_queue = stack<Cell*, std::vector<Cell*>>(cells);
    for (auto* cell : cells) cell->init_async();
}

void CellProcessor::layer_init(
    const std::vector<Cell *> &cells,
    ResourceCollector<SDFPath, Cell>& sdf_collector, ResourceCollector<Transition, Wire>& input_data_collector
) {
    for (auto* cell : cells) cell->init(sdf_collector, input_data_collector, overflow_collector, s_overflow_collector);
}

void CellProcessor::set_ptrs(SDFPath *sdf, Transition *input_data) {
    device_sdf = sdf; device_input_data = input_data;
}

bool CellProcessor::run() {
    if (has_unfinished) return false;
    else if (!has_unfinished and job_queue.empty()) return true;

    cout << "size of job_queue:" << job_queue.size() << endl;
    processing_cells.clear();
    s_slice_info_collector.reset(); s_delay_info_collector.reset(); s_length_collector.reset();
    output_data_collector.reset(); output_size_collector.reset();

    for (int i = 0; i < N_CELL_PARALLEL; i++) {
        if (job_queue.empty()) break;
        auto* cell = job_queue.top(); processing_cells.insert(cell);
        cell->prepare_resource(
                session_id, resource_buffer,
                output_data_collector, output_size_collector,
                s_slice_info_collector, s_delay_info_collector, s_length_collector
        );
        if (cell->finished()) job_queue.pop();
    }
    batch_data.set(resource_buffer, stream); resource_buffer.clear();

    auto* device_s_slice_infos = s_slice_info_collector.get_device(stream);
    auto* device_s_delay_infos = s_delay_info_collector.get_device(stream);
    auto* device_overflow = overflow_collector.get_device(stream);
    auto* device_s_overflow = s_overflow_collector.get_device(stream);
    auto* device_output_data = output_data_collector.get_device(stream);
    auto* device_s_lengths = s_length_collector.get_device(stream);
    auto* device_sizes = output_size_collector.get_device(stream);
    slice_kernel<<<N_CELL_PARALLEL, 1, 0, stream>>>(
            batch_data, device_input_data, device_s_slice_infos, device_s_overflow
    );
    cudaStreamSynchronize(stream);
    cout << "finish slice\n";
    stepping_kernel<<<N_CELL_PARALLEL, N_STIMULI_PARALLEL, 0, stream>>>(
            batch_data, device_input_data, device_output_data, device_s_slice_infos, device_s_delay_infos, device_overflow
    );
    cudaStreamSynchronize(stream);
    cout << "finish stepping\n";
    compute_delay_kernel<<<N_CELL_PARALLEL, N_STIMULI_PARALLEL, 0, stream>>>(
            batch_data, device_sdf, device_output_data,
            device_s_delay_infos, device_s_lengths
    );
    cudaStreamSynchronize(stream);
    cout << "finish delay\n";
    resolve_collision_kernel<<<N_CELL_PARALLEL, 1, 0, stream>>>(
            batch_data, device_output_data, device_sizes,
            device_s_lengths
    );
    cudaStreamSynchronize(stream);
    cout << "finish collision\n";
    host_output_data = output_data_collector.get_host(stream);
    host_sizes = output_size_collector.get_host(stream);
    host_overflows = overflow_collector.get_host(stream);
    host_s_overflows = s_overflow_collector.get_host(stream);

    cudaStreamAddCallback(stream, CellProcessor::post_process, (void*) this, 0);
    has_unfinished = true;
    return false;
}

CUDART_CB void CellProcessor::post_process(cudaStream_t stream, cudaError_t status, void* userData) {
    auto* processor = static_cast<CellProcessor*>(userData);

    unordered_set<Cell*> non_overflow_cells, finished_cells;
    for (auto* cell : processor->processing_cells) {
        bool finished = cell->finished();
        bool overflow = cell->handle_overflow(processor->host_overflows, processor->host_s_overflows);
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
