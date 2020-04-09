import pytest
from typing import List

from numba import cuda
import numpy as np

from ..data_structures import ANDGate, batch_data


def make_mock_inputs(stimuli_size: int, n_inputs: int) -> List[batch_data]:
    return [
        batch_data(
            values=np.random.choice(
                np.array([0, 1, 2, 3], dtype=np.byte),
                size=stimuli_size,
            ),
            timestamps=np.zeros(stimuli_size),
        )
        for _ in range(n_inputs)
    ]


def cpu_and(inputs: List[batch_data]) -> batch_data:
    return batch_data(
        values=inputs[0].values,
        timestamps=inputs[0].timestamps,
    )

#
# @cuda.jit
# def gpu_and(inputs: List[batch_data], outputs: batch_data):
#     outputs.values[...] = inputs[0].values
#     outputs.timestamps[...] = inputs[0].timestamps


@pytest.mark.parametrize(
    "gate_cls, cpu_fn", [
        (ANDGate, cpu_and)
    ]
)
def test_builtin_gates(gate_cls, cpu_fn):
    STIMULI_SIZE = 100
    N_INPUTS = 5
    mock_inputs = make_mock_inputs(STIMULI_SIZE, N_INPUTS)

    device_mock_inputs = [batch_data(values=cuda.to_device(i.values), timestamps=i.timestamps) for i in mock_inputs]
    outputs = batch_data(
        values=cuda.device_array(shape=(STIMULI_SIZE,), dtype=np.byte),
        timestamps=cuda.device_array(shape=(STIMULI_SIZE,), dtype=np.int),
    )

    gate = gate_cls()
    gate.compute[1, 1](device_mock_inputs, outputs)
    host_outputs = batch_data(values=outputs.values.copy_to_host(), timestamps=outputs.timestamps.copy_to_host())
    expected_ans = cpu_fn(mock_inputs)

    np.testing.assert_equal(expected_ans.values, host_outputs.values)
    np.testing.assert_equal(expected_ans.timestamps, host_outputs.timestamps)
