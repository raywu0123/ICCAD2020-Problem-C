from typing import List, Tuple, Dict


BIT_INDEX_TYPE = int
SINGLE_BIT_INDEX = 0
WireKey = Tuple[str, BIT_INDEX_TYPE]

BITWIDTH_TYPE = Tuple[int, int]

UNDEF_USAGE_BITWIDTH = (-1, -1)
CELL_ID_TYPE, PIN_ID_TYPE = str, str
ASSIGNMENT_MAP_TYPE = Dict[
    PIN_ID_TYPE,
    List[Tuple[CELL_ID_TYPE, PIN_ID_TYPE, BITWIDTH_TYPE]],
]
