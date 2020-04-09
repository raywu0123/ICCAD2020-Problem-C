from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict

from numba import cuda


batch_data = namedtuple('batch_data', ['values', 'timestamps'])


class Gate(ABC):

    @staticmethod
    @abstractmethod
    def compute(inputs, outputs):
        pass


class ANDGate(Gate):

    @staticmethod
    @cuda.jit
    def compute(inputs, outputs):
        outputs.values[...] = inputs[0].values
        outputs.timestamps[...] = inputs[0].timestamps


class ORGate(Gate):

    def compute(self, inputs, outputs):
        pass


class NOTGate(Gate):

    def compute(self, inputs, outputs):
        pass


class XORGate(Gate):

    def compute(self, inputs, outputs):
        pass


class XNORGate(Gate):

    def compute(self, inputs, outputs):
        pass


class NANDGate(Gate):

    def compute(self, inputs, outputs):
        pass


class NORGate(Gate):

    def compute(self, inputs, outputs):
        pass


class BUFGate(Gate):

    def compute(self, inputs, outputs):
        pass


class Primitive(Gate):

    def __init__(self, declares, table):
        self.table = table

    def compute(self, inputs, outputs):
        pass


class Module(Gate):

    def __init__(self, input_names, output_names, gates):
        pass

    def compute(self, inputs: Dict[str, batch_data], outputs: Dict[str, batch_data]):
        pass
