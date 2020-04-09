from typing import Set
from graph_preprocessing.constants import WireKey
from graph_preprocessing.utils import make_wire_key


class VCD:

    def __init__(self):
        self.timescale = None
        self.symbol_to_wirespec = {}
        self.dumps = {}

    def summary(self):
        print("Summary of Input VCD:")
        print(f"Timescale: {self.timescale}")
        print(f"Num dumps: {len(self.dumps)}")

    def read(self, path: str):
        with open(path, 'r') as f_in:
            for line in f_in:
                if line.startswith('$timescale'):
                    self.timescale = line.split(' ')[1]
                    break

            for line in f_in:
                if line.startswith('$dumpvars'):
                    break
                elif line.startswith('$var'):
                    split = line.split(' ')
                    symbol, wirename = split[3:5]
                    bitwidth = self.parse_bitwidth(split[5]) if len(split) == 7 else (0, 0)
                    self.symbol_to_wirespec[symbol] = (wirename, bitwidth)

            current_bucket = None
            for line in f_in:
                line = line.strip('\n')
                if line[0] == '#':
                    timestamp = int(line[1:])
                    if timestamp in self.dumps:
                        raise ValueError(f"Duplicate timestamps: {timestamp}")
                    self.dumps[timestamp] = []
                    current_bucket = self.dumps[timestamp]
                else:
                    if ' ' not in line:
                        # single bit
                        current_bucket.append((line[0], line[1:]))
                    else:
                        # bus
                        split = line.split(' ')
                        assert split[0][0] == 'b'
                        current_bucket.append((split[0][1:], split[1]))

    @staticmethod
    def parse_bitwidth(s: str):
        return tuple([int(i) for i in s.strip('[]').split(':')])

    def get_wire_keys(self) -> Set[WireKey]:
        return {
            make_wire_key(wire_name, bit_index)
            for wire_name, bitwidth in self.symbol_to_wirespec.values()
            for bit_index in range(min(bitwidth), max(bitwidth) + 1)
        }

    def __next__(self):
        pass
