import os

from pyparsing import (
    Word,
    Keyword,
    printables,
    quotedString,
    Group,
    Suppress,
    ParserElement,
)

from .parser_common import (
    enclosedExpr,
    variable,
    floatnum,
    posnegedge_keyword,
)


class SDFParser:

    def __init__(self):
        self.cell_grammar = self.get_cell_grammar()

    @staticmethod
    def get_cell_grammar() -> ParserElement:
        cell_type = enclosedExpr(Suppress(Keyword("CELLTYPE")) + quotedString)

        instance = enclosedExpr(Suppress("INSTANCE") + variable)
        null_instance = enclosedExpr(Suppress(Keyword("INSTANCE")))

        partial_path = enclosedExpr(floatnum + Suppress(':') + floatnum + Suppress(':') + floatnum)
        path_input = enclosedExpr(Group(posnegedge_keyword + variable)) | variable
        iopath = enclosedExpr(Suppress('iopath') + path_input + variable + Group(partial_path[1, 2]))

        interconnect = enclosedExpr(Keyword("INTERCONNECT") + Word(printables) * 2 + partial_path)

        delay = enclosedExpr(
            Suppress(Keyword("DELAY")) + enclosedExpr(
                Suppress(Keyword("ABSOLUTE")) + (iopath[1, ...] | interconnect)
            )
        )

        cell = enclosedExpr(
            Suppress(Keyword("CELL")) + cell_type('type') + (instance('name') | null_instance) + delay('delay')
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

    def parser_string(self, s: str):
        return dict(self.cell_grammar.parseString(s))

    def read_file(self, path: str):
        if not os.path.isfile(path):
            raise ValueError(f"Vlib file {path} don't exist")

        cells = []
        with open(path, 'r') as f:
            header, celldef = self.read_header(f)
            for line in f:
                if line.rstrip('\n ') == '(CELL':
                    result = self.parser_string(celldef.strip(' \n'))
                    cells.append(result)
                    celldef = line
                else:
                    celldef += line
            result = self.parser_string(celldef.strip(' \n'))
            cells.append(result)
        return header, cells
