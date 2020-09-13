#ifndef ICCAD2020_SIMULATOR_H
#define ICCAD2020_SIMULATOR_H

#include <vector>
#include <stack>

#include "circuit_model/circuit.h"
#include "simulation_result.h"
#include "constants.h"

__device__ __host__ void slice_waveforms(
    SliceInfo* s_slice_infos,
    Transition* all_input_data, InputData* data, const CAPACITY_TYPE& capacity,
    const NUM_ARG_TYPE& num_wires, bool* overflow_ptr
);

__host__ __device__ void stepping_algorithm(
    Transition** input_data, const unsigned int* sizes,
    DelayInfo* s_delay_infos,
    Transition** output_data,
    const ModuleSpec* module_spec,
    const CAPACITY_TYPE& capacity,
    bool* overflow_ptr,
    bool verbose = false
);

__host__ __device__ int lookup_delay(
    NUM_ARG_TYPE, NUM_ARG_TYPE, EdgeTypes, EdgeTypes,
    const SDFPath*, const unsigned int&
);

__host__ __device__ void compute_delay(
    Transition**, const CAPACITY_TYPE& capacity, DelayInfo*,
    const NUM_ARG_TYPE&, const NUM_ARG_TYPE&,
    const SDFPath* sdf_paths, const unsigned int& sdf_num_rows,
    CAPACITY_TYPE* lengths, bool verbose = false
);




class Simulator {
public:
    explicit Simulator(Circuit& c): circuit(c) {};
    void run();

private:

    Circuit& circuit;
};

class CellProcessor {
public:
    CellProcessor();
    ~CellProcessor();

    static void layer_init_async(CellProcessor& processor, const std::vector<Cell*>& cells);
    void layer_init(
        const std::vector<Cell*>& cells,
        ResourceCollector<SDFPath, Cell>& sdf_collector, ResourceCollector<Transition, Wire>& input_data_collector
    );
    void set_ptrs(SDFPath* sdf, Transition* input_data);
    bool run();
    static void CUDART_CB post_process(cudaStream_t stream, cudaError_t status, void* processor);

    SDFPath* device_sdf = nullptr;
    Transition* device_input_data = nullptr;

    Transition* host_output_data = nullptr;
    unsigned int* host_sizes = nullptr;
    bool *host_overflows = nullptr, *host_s_overflows = nullptr;

    std::stack<Cell*, std::vector<Cell*>> job_queue;
    std::unordered_set<Cell*> processing_cells;

    ResourceBuffer resource_buffer;
    BatchResource batch_data{};

    OutputCollector<SliceInfo> s_slice_info_collector{N_CELL_PARALLEL * N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT};
    OutputCollector<DelayInfo> s_delay_info_collector{N_CELL_PARALLEL * N_STIMULI_PARALLEL * INITIAL_CAPACITY};
    OutputCollector<CAPACITY_TYPE> s_length_collector{N_CELL_PARALLEL * N_STIMULI_PARALLEL * MAX_NUM_MODULE_OUTPUT};

    OutputCollector<Transition> output_data_collector{N_CELL_PARALLEL * N_STIMULI_PARALLEL * INITIAL_CAPACITY * MAX_NUM_MODULE_OUTPUT};
    OutputCollector<unsigned int> output_size_collector{N_CELL_PARALLEL * MAX_NUM_MODULE_OUTPUT};
    OutputCollector<bool> overflow_collector, s_overflow_collector;

    cudaStream_t stream;

    bool has_unfinished = false;
    int session_id = 0;
};

#endif
