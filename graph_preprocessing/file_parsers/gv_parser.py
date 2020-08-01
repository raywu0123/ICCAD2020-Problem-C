from collections import namedtuple
from typing import Tuple, List, Set, Union
from itertools import chain

from ..constants import SINGLE_BIT_INDEX

CellInfo = namedtuple('CellInfo', ['type', 'name', 'args'])
GVInfo = namedtuple('GVInfo', ['wire', 'assign', 'cell', 'design_name'])
WireInfo = namedtuple('WireInfo', ['id', 'bitwidth'])
ArgInfo = namedtuple("ArgInfo", ['pin_name', 'wire_info'])


class GVParser:

    @classmethod
    def read_file(cls, path, valid_cell_types: Set[str] = None) -> GVInfo:
        with open(path, 'r') as fin:
            content = fin.read()

        lines = content.split(';')
        lines = [line.strip(' \n\t') for line in lines]

        head = lines[0][lines[0].find('module') + len('module '):]
        design_name = head[:head.find('(')].strip()

        buckets = {
            'wire': [],
            'assign': [],
            'cell': [],
        }

        for line in lines[1:-1]:
            if line.startswith('input') or line.startswith('output') or line.startswith('wire'):
                buckets['wire'].append(line)
            elif line.startswith('assign'):
                buckets['assign'].append(line)
            else:
                buckets['cell'].append(line)

        return GVInfo(
            wire=list(chain(*[cls.parse_wire(line) for line in buckets['wire']])),
            assign=[cls.parse_assign(line) for line in buckets['assign']],
            cell=list(filter(
                lambda x: x is not None,
                [cls.parse_cell(line, valid_cell_types) for line in buckets['cell']]
            )),
            design_name=design_name
        )

    @staticmethod
    def parse_wire(line: str) -> List[Tuple[str, WireInfo]]:
        split = line.split()
        wire_type = split[0]

        if '[' in split[1]:
            bitwidth = split[1].lstrip('[').rstrip(']').split(':')
            bitwidth = (int(bitwidth[0]), int(bitwidth[1]))
            wire_id = split[2]
            wire_list = [(wire_type, WireInfo(id=wire_id, bitwidth=bitwidth))]
        else:
            wire_list = [(wire_type, WireInfo(id=term, bitwidth=None)) for term in ''.join(split[1:]).split(',')]
        return wire_list

    @classmethod
    def parse_assign(cls, line: str) -> Tuple[Tuple[str, int], Tuple[str, int]]:
        line = line[len('assign'):]
        split = line.split('=')
        return cls.convert_wire_term(split[0]), cls.convert_wire_term(split[1])

    @staticmethod
    def convert_wire_term(s: str) -> Tuple[str, int]:
        term = s.strip(' ')
        if '[' in term:
            split = term.split('[')
            name = split[0]
            index = int(split[1].strip(']'))
        else:
            index = SINGLE_BIT_INDEX
            name = term
        return name, index

    @classmethod
    def parse_cell(cls, line: str, valid_cell_types: Set[str]) -> Union[CellInfo, None]:
        space_index, left_bracket_index = line.find(' '), line.find('(')
        cell_type, cell_name = line[:space_index], line[space_index + 1:left_bracket_index]
        if valid_cell_types is not None and cell_type not in valid_cell_types:
            return None

        arg_list = line[left_bracket_index + 1:-1].replace('\n', '').strip().split(',')
        arg_list = [term.lstrip('\n .').rstrip('\n) ').split('(') for term in arg_list]
        arg_list = [(term[0].strip(), term[1].strip()) for term in arg_list]
        arg_list = [
            ArgInfo(pin_name=term[0], wire_info=cls.convert_wire_term(term[1]))
            for term in arg_list
        ]
        return CellInfo(type=cell_type, name=cell_name, args=arg_list)
