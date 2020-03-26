import os
import re

from pyparsing import (
    Word,
    printables,
    quotedString,
    Group,
    Suppress,
    ParserElement,
    pyparsing_common,
)
from tqdm import tqdm

from .parser_common import (
    enclosedExpr,
    variable,
    posnegedge_keyword,
    make_keyword,
)


class SDFParser:

    def __init__(self):
        self.cell_grammar = self.get_cell_grammar()

    @staticmethod
    def get_cell_grammar() -> ParserElement:
        cell_type = enclosedExpr(Suppress(make_keyword("CELLTYPE")) + quotedString)

        instance = enclosedExpr(Suppress("INSTANCE") + variable)
        null_instance = enclosedExpr(Suppress(make_keyword("INSTANCE")))

        float_num = pyparsing_common.real
        colon = Suppress(':')
        partial_path = enclosedExpr(float_num + colon + float_num + colon + float_num)
        path_input = variable | enclosedExpr(Group(posnegedge_keyword + variable))

        iopath_kw = Suppress('iopath')
        iopath = enclosedExpr(iopath_kw + path_input + variable + Group(partial_path[1, 2]))

        interconnect_kw = make_keyword("INTERCONNECT")
        interconnect = enclosedExpr(interconnect_kw + Word(printables) * 2 + partial_path)

        delay_kw = Suppress(make_keyword("DELAY"))
        absolute_kw = Suppress(make_keyword("ABSOLUTE"))
        delay = enclosedExpr(
            delay_kw + enclosedExpr(
                absolute_kw + (iopath[1, ...] | interconnect)
            )
        )

        cell_kw = Suppress(make_keyword("CELL"))
        cell = enclosedExpr(
            cell_kw + cell_type('type') + (instance('name') | null_instance) + delay('delay')
        )
        return cell

    @staticmethod
    def read_header(f) -> (str, str):
        header = {}
        last_line = ''
        for line in f:
            if line.startswith('(DELAYFILE'):
                continue

            line = line.rstrip('\n ')
            if line != '(CELL':
                line = line.strip('()\n ')
                key = line.split()[0]
                header[key] = line[len(key):].strip('" ')
            else:
                last_line = line + '\n'
                break
        return header, last_line

    def parse_string(self, s: str):
        return dict(self.cell_grammar.parseString(s))

    def read_file(self, path: str):
        if not os.path.isfile(path):
            raise ValueError(f"SDF file {path} don't exist")

        with open(path, 'r') as f:
            header, last_line = self.read_header(f)
            celldefs = last_line + f.read()

        indices = [m.start() for m in re.finditer(r"\(CELL[^T]", celldefs)]
        indices.append(-1)

        cells = [
            self.parse_string(celldefs[start_idx: end_idx])
            for start_idx, end_idx in tqdm(zip(indices[:-1], indices[1:]), total=len(indices) - 1)
        ]
        return header, cells
