#ifndef ICCAD2020_JOB_H
#define ICCAD2020_JOB_H

#include "circuit_model/wire.h"
#include <queue>

struct JobHandle {
    virtual ~JobHandle() = default;
    virtual void handle_overflow(unsigned int) {};
    virtual Data prepare(const cudaStream_t&) { return Data{}; };
    virtual void finish(const cudaStream_t&) {};

    Data device_data;
};

struct InputJobHandle : public JobHandle {
    // life cycle : prepare -> (prepare) * N -> finish

    InputJobHandle(Transition* ptr, unsigned int size);
    Data prepare(const cudaStream_t&) override;
    void finish(const cudaStream_t&) override;

    Transition* transitions;
    unsigned int size;
};

struct OutputJobHandle : public JobHandle {
    // life cycle : prepare -> (handle overflow -> prepare) * N -> finish

    explicit OutputJobHandle(PinnedMemoryVector<Transition>&);
    Data prepare(const cudaStream_t&) override;
    void finish(const cudaStream_t&) override;
    void handle_overflow(unsigned int new_capacity) override;

    PinnedMemoryVector<Transition>& buffer;
    unsigned int capacity = INITIAL_CAPACITY;
};


struct WrappedWire {
    explicit WrappedWire(Wire* w) : wire(w) {}

    virtual void free() = 0;
    virtual JobHandle* get_job_handle(unsigned int idx) = 0;

    Wire* wire;
};

struct OutputWire : public WrappedWire {
    explicit OutputWire(Wire* w) : WrappedWire(w) {};

    void finish();
    void free() override;
    JobHandle* get_job_handle(unsigned int idx) override;

    void set_schedule_size(unsigned int size);

    PinnedMemoryVector<PinnedMemoryVector<Transition>> buffers;
};


struct InputWire : public WrappedWire{
    explicit InputWire(Wire* w) : WrappedWire(w) {};
    void free() override;
    JobHandle* get_job_handle(unsigned int idx) override;

    void push_back_schedule_index(unsigned int i);
    std::vector<unsigned int> bucket_index_schedule{ 0 };
};

class Job {
public:
//    owns job_handles
    Job(const ModuleSpec*, const SDFSpec*, unsigned int num_args, std::vector<JobHandle*> handles);
    ~Job();
    void init(const cudaStream_t& stream);
    void push_resource(ResourceBuffer&, const cudaStream_t& stream);
    void finish(const cudaStream_t& stream);
    void handle_overflow();

friend class BatchJob;
private:
    bool* overflow_ptr = nullptr;
    const ModuleSpec* module_spec;
    const SDFSpec* sdf_spec;
    unsigned int num_args;
    std::vector<JobHandle*> handles;
    unsigned int capacity = INITIAL_CAPACITY;
};

class BatchJob {
public:
    BatchJob(const std::vector<Job*>& jobs, const cudaStream_t& stream);
    void init();
    void execute();
    void push_unfinished(std::queue<Job*>& job_queue) const;
    void finish() const;

    const std::vector<Job*>& jobs;
    const cudaStream_t& stream;
    bool* overflows = nullptr;
};

#endif //ICCAD2020_JOB_H
