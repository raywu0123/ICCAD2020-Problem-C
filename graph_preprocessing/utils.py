from typing import Tuple

from .constants import UNDEF_USAGE_BITWIDTH, BITWIDTH_TYPE


def is_single_bit(bitwidth: BITWIDTH_TYPE):
    return bitwidth[0] == bitwidth[1]


def extract_bitwidth(s: str) -> (str, Tuple[int, int]):
    split = s.split('[')
    if len(split) == 1:
        return s, UNDEF_USAGE_BITWIDTH
    elif len(split) == 2:
        b = split[1].strip(']').split(":")
        if len(b) == 1:
            return split[0], (int(b[0]), int(b[0]))
        elif len(b) == 2:
            return split[0], (int(b[0]), int(b[1]))
        else:
            raise ValueError(f"Unrecognized format: {s}")
    else:
        raise ValueError(f"Invalid argument: {s}")
