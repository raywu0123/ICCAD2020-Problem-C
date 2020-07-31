#include <thread>

#include "simulator/simulator.h"
#include "include/progress_bar.h"

using namespace std;


void Simulator::run() {
    cout << "| Status: Running Simulation... " << endl;

    size_t new_heap_size = N_CELL_PARALLEL * N_STIMULI_PARALLEL * INITIAL_CAPACITY * 8
            * (sizeof(Timestamp) + sizeof(DelayInfo) + sizeof(Values) * MAX_NUM_MODULE_ARGS);
    cudaErrorCheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size));
    cout << "| Adjusted heap size to be " << new_heap_size  << " bytes" << endl;

    unsigned int num_layers = circuit.cell_schedule.size();
    cout << "| Total " << num_layers << " layers" << endl;

    ProgressBar progress_bar(num_layers);
    for (unsigned int i_layer = 0; i_layer < num_layers; i_layer++) {
        const auto& schedule_layer = circuit.cell_schedule[i_layer];

        queue<Job*> job_queue;
        mutex m;
        for (auto* cell : schedule_layer) cell->push_jobs(job_queue);
        auto jobs = job_queue;
        vector<thread> threads;
        for (int i = 0; i < N_THREAD; ++i)
            threads.emplace_back(worker, std::ref(m), std::ref(job_queue));

        for (auto& thread : threads) thread.join();

        while(not jobs.empty()) delete jobs.front(), jobs.pop();
        for (auto* cell : schedule_layer) cell->finish();

        progress_bar.Progressed(i_layer + 1);
    }
    cout << endl;
}

void Simulator::worker(mutex &m, queue<Job*>& job_queue) {
    unique_lock<mutex> lock(m); lock.unlock();

    // to reuse streams, let thread_fn create and destroy
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    while (true) {
        lock.lock();
        if (job_queue.empty()) break;

        vector<Job*> acquired_jobs; acquired_jobs.reserve(N_CELL_PER_THREAD);
        while (acquired_jobs.size() < N_CELL_PER_THREAD and not job_queue.empty()) {
            auto* job = job_queue.front(); job_queue.pop();
            acquired_jobs.push_back(job);
        }
        lock.unlock();

        BatchJob batch_job(acquired_jobs, stream); batch_job.init();
        batch_job.execute();

        lock.lock();
        batch_job.push_unfinished(job_queue);
        lock.unlock();

        batch_job.finish();
    }
    cudaStreamDestroy(stream);
}
