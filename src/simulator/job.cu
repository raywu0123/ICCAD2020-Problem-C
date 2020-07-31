#include <utility>
#include <vector>
#include <thread>

#include "job.h"
#include "simulator/device_functions.h"
#include "utils.h"

using namespace std;

void InputWire::push_back_schedule_index(unsigned int i) {
    if (i > wire->bucket.size())
        throw std::runtime_error("Schedule index out of range.");
    if (not bucket_index_schedule.empty() and i < bucket_index_schedule.back())
        throw std::runtime_error("Schedule index in incorrect order.");
    bucket_index_schedule.push_back(i);
}

Job::Job(
    const ModuleSpec* module_spec, const SDFSpec* sdf_spec,
    unsigned int num_args, vector<JobHandle*> handles
): module_spec(module_spec), sdf_spec(sdf_spec), num_args(num_args), handles(std::move(handles)) {}

Job::~Job() {
    for (auto* handle : handles) delete handle;
}

void Job::init(const cudaStream_t &stream) {
    if (overflow_ptr == nullptr) cudaMalloc((void**) &overflow_ptr, sizeof(bool));
}

void Job::handle_overflow() {
    capacity *= 2;
    for (auto& handle : handles) handle->handle_overflow(capacity);
}

//bool Job::execute(const cudaStream_t& stream) {
//    PinnedMemoryVector<Data> data_vec; data_vec.reserve(num_args);
//    for (auto& handle : handles) data_vec.push_back(handle->prepare(stream));
//
//    Data* data_list;
//    cudaMalloc((void**) &data_list, sizeof(Data) * data_vec.size());
//    cudaMemcpyAsync(data_list, data_vec.data(), sizeof(Data) * data_vec.size(), cudaMemcpyHostToDevice, stream);
//
//    cudaMemsetAsync(overflow_ptr, 0, sizeof(bool), stream);
//    simulate_module<<<1, N_STIMULI_PARALLEL, 0, stream>>>(
//        module_spec, sdf_spec,
//        data_list, capacity,
//        overflow_ptr
//    );
//
//    bool* overflow;
//    cudaMallocHost((void**) &overflow, sizeof(bool));
//    cudaMemcpyAsync(overflow, overflow_ptr, sizeof(bool), cudaMemcpyDeviceToHost, stream);
//    cudaStreamSynchronize(stream);
//
//    if (*overflow) {
//        capacity *= 2;
//        for (auto& handle : handles) handle->handle_overflow(capacity);
//    }
//    return not *overflow;
//}
//
void Job::finish(const cudaStream_t &stream) {
    cudaFree(overflow_ptr);
    for (auto& handle : handles) handle->finish(stream);
}

void Job::push_resource(ResourceBuffer& resource_buffer, const cudaStream_t& stream) {
    cudaMemsetAsync(overflow_ptr, 0, sizeof(bool), stream);

    resource_buffer.module_specs.push_back(module_spec);
    resource_buffer.sdf_specs.push_back(sdf_spec);
    resource_buffer.overflow_ptrs.push_back(overflow_ptr);
    resource_buffer.capacities.push_back(capacity);
    for (auto& handle : handles) resource_buffer.data_list.push_back(handle->prepare(stream));
    resource_buffer.data_list.resize(resource_buffer.module_specs.size() * MAX_NUM_MODULE_ARGS);
}

JobHandle* InputWire::get_job_handle(unsigned int idx) {
    auto start_idx = bucket_index_schedule[idx];
    if (start_idx != 0) start_idx--;
    const auto& end_index = bucket_index_schedule[idx + 1];
    auto size = end_index - start_idx;
    return new InputJobHandle(wire->bucket.data() + start_idx, size);
}

void InputWire::free() {
    vector<unsigned int>().swap(bucket_index_schedule);
}

InputJobHandle::InputJobHandle(Transition *ptr, unsigned int size) : transitions(ptr), size(size) {}

Data InputJobHandle::prepare(const cudaStream_t& stream) {
    if (device_data.transitions != nullptr) return device_data; // no need to reallocate

    cudaMalloc((void**) &device_data.transitions, sizeof(Transition) * INITIAL_CAPACITY * N_STIMULI_PARALLEL);
    cudaMemcpyAsync(device_data.transitions, transitions, sizeof(Transition) * size, cudaMemcpyHostToDevice, stream);
    return device_data;
}

void InputJobHandle::finish(const cudaStream_t& stream) {
    cudaFree(device_data.transitions);
}

OutputJobHandle::OutputJobHandle(PinnedMemoryVector<Transition>& buffer) : buffer(buffer) {}

Data OutputJobHandle::prepare(const cudaStream_t& stream) {
    // always allocate
    auto transitions_size = sizeof(Transition) * capacity * N_STIMULI_PARALLEL;
    cudaMalloc((void**) &device_data.transitions, transitions_size);
    cudaMemsetAsync(device_data.transitions, 0, transitions_size, stream);

    // no need to reallocate size indicator
    if (device_data.size == nullptr) cudaMalloc((void**) &device_data.size, sizeof(unsigned int));

    // always reset
    cudaMemsetAsync(device_data.size, 0, sizeof(unsigned int), stream);
    return device_data;
}

void OutputJobHandle::finish(const cudaStream_t& stream) {
    unsigned int* size_ptr; cudaMallocHost((void**) &size_ptr, sizeof(unsigned int));
    cudaMemcpyAsync(size_ptr, device_data.size, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    buffer.resize(*size_ptr);

    cudaMemcpyAsync(buffer.data(), device_data.transitions, sizeof(Transition) * *size_ptr, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(device_data.transitions);
    cudaFree(device_data.size);
}

void OutputJobHandle::handle_overflow(unsigned int new_capacity) {
    capacity = new_capacity;
    cudaFree(device_data.transitions);
}

JobHandle* OutputWire::get_job_handle(unsigned int idx) {
    return new OutputJobHandle(buffers[idx]);
}

void OutputWire::free() {
    PinnedMemoryVector<PinnedMemoryVector<Transition>>().swap(buffers);
}

void OutputWire::finish() {
    unsigned int sum_length = 0;
    for (const auto& buffer : buffers) sum_length += buffer.size();
    auto& bucket = wire->bucket;
    bucket.reserve(sum_length);

    for (const auto& buffer : buffers) {
        unsigned int offset = 0;
        if (not bucket.empty() and not buffer.empty() and bucket.back().value == buffer.front().value) offset = 1;
        const auto& buffer_size = buffer.size();
        for (unsigned int i = offset; i < buffer_size; ++i) bucket.push_back(buffer[i]);
    }
}

void OutputWire::set_schedule_size(unsigned int size) {
    buffers.resize(size);
}

BatchJob::BatchJob(const std::vector<Job*>& jobs, const cudaStream_t& stream) : jobs(jobs), stream(stream) {}

void BatchJob::init() {
    cudaMallocHost((void**) &overflows, sizeof(bool) * jobs.size());
}

void BatchJob::finish() const {
    cudaFreeHost(overflows);
}

void BatchJob::execute() {
    for (auto& job : jobs) job->init(stream);

    const auto& num_jobs = jobs.size();
    ResourceBuffer resource_buffer(num_jobs);
    for (auto& job : jobs) job->push_resource(resource_buffer, stream);

    BatchResource batch_resource{}; batch_resource.init(resource_buffer, stream);
    simulate_batch<<<num_jobs, N_STIMULI_PARALLEL, 0, stream>>>(batch_resource);

    resource_buffer.get_overflows(overflows, stream);
    this_thread::yield();
    cudaStreamSynchronize(stream);
    batch_resource.finish();

    for (int i = 0; i < num_jobs; ++i) {
        if (overflows[i]) jobs[i]->handle_overflow();
        else jobs[i]->finish(stream);
    }
}

void BatchJob::push_unfinished(queue<Job*>& job_queue) const {
    for (int i = 0; i < jobs.size(); ++i) if (overflows[i]) job_queue.push(jobs[i]);
}
