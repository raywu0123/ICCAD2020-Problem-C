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

void Wire::store_to_bucket(const std::vector<Data>& data_list, Transition* output_data, unsigned int* sizes) {
    for(const auto& data : data_list) bucket.push_back(output_data + data.transition_offset, sizes[data.size_offset]);
}

void Wire::set_drived() {
    bucket.transitions[0].value = Values::X;
}

void Wire::emplace_transition(const Timestamp &t, char r) {
    bucket.emplace_transition(t, r);
}

void Wire::to_device(ResourceCollector<Transition>& input_data_collector) {
    ref_count++;
    if (ref_count > 1) return;
    offset = input_data_collector.push(bucket.transitions);
}

void Wire::free_device() {
    ref_count--;
}

bool ConstantWire::store_to_bucket_warning_flag = false;
bool ConstantWire::emplace_transition_warning_flag = false;
