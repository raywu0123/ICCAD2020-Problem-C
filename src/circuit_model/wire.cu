#include <cassert>

#include "wire.h"
#include "utils.h"

using namespace std;

Wire::Wire(const WireInfo& wire_info) {
    wire_infos.push_back(wire_info);
}

// memory pattern
//      0                   1   2   3  ...    C
// 0   previous_transition t00 t01 t02 ... t0(c-1)
// 1   t0(c-1)             t10 t11 t12
// 2   t1(c-1)
// .
// .
// N-1


void Wire::assign(const Wire& other_wire) {
    wire_infos.insert(wire_infos.end(), other_wire.wire_infos.begin(), other_wire.wire_infos.end());
}

void Wire::store_to_bucket(const std::vector<Data>& storage) {
    for (const auto& data : storage) { bucket.push_back(data.transitions, *data.size); }
}

void Wire::set_drived() {
    bucket.transitions[0].value = Values::X;
}

void Wire::emplace_transition(const Timestamp &t, char r) {
    bucket.emplace_transition(t, r);
}

void Wire::to_device(cudaStream_t stream) {
    ref_count++;
    if (device_ptr != nullptr) return;

    auto size = sizeof(Transition) * bucket.size();
    cudaMallocHost((void**) &pinned_host_ptr, size);
    memcpy(pinned_host_ptr, bucket.transitions.data(), size);

    cudaMalloc((void**) &device_ptr, size);
    cudaMemcpyAsync(device_ptr, pinned_host_ptr, size, cudaMemcpyHostToDevice, stream);
}

void Wire::free_device() {
    ref_count--;
    if (ref_count > 0) return;

    cudaFree(device_ptr); cudaFreeHost(pinned_host_ptr);
    device_ptr = nullptr; pinned_host_ptr = nullptr;
}

bool ConstantWire::store_to_bucket_warning_flag = false;
bool ConstantWire::emplace_transition_warning_flag = false;
