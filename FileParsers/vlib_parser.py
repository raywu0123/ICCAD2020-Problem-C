from typing import List
from pprint import pprint
from functools import reduce

from pyparsing import (
    Keyword,
    Word,
    Group,
    Suppress,
    delimitedList,
    nums,
    ParserElement,
    ParseResults,
    restOfLine,
    SkipTo,
)

from .parser_common import (
    enclosedExpr,
    variable_list,
    variable,
    timescale,
    posnegedge_keyword,
)


class VLibParser:

    def __init__(self):
        self.modules = []
        self.timescale = None

        self.module_spec = {}
        self._grammar = self.get_grammar()

    @property
    def grammar(self) -> ParserElement:
        return self._grammar

    def get_primitive_grammar(self, variable_declare: ParserElement):
        basic_table_token = Word("01?x-*", max=1)
        bracket_table_token = Group("(" + Word("01?x-", max=2) + ")")
        table_token = basic_table_token | bracket_table_token
        table_line = Group(delimitedList(Group(table_token[1, ...]), delim=":")) + Suppress(';\n')
        table_declare = enclosedExpr(
            content=table_line[1, ...],
            opener='table',
            closer='endtable',
        )
        table_declare.setParseAction(self.add_table)
        primitive_declare = enclosedExpr(
            variable + enclosedExpr(variable_list) + Suppress(";") +
            Group(variable_declare)[...] + table_declare,
            opener="primitive",
            closer='endprimitive',
        )
        primitive_declare.setParseAction(self.add_cell)
        return primitive_declare

    def get_module_grammar(self, variable_declare: ParserElement):
        specify_path = enclosedExpr(
            Group(
                posnegedge_keyword + variable + Suppress('=>') + variable
            ),
        )
        specify_timing = enclosedExpr(Group(delimitedList(Word(nums))))
        specify_line = Group(specify_path + Suppress("=") + specify_timing) + Suppress(';\n')
        specify = enclosedExpr(
            content=specify_line[1, ...],
            opener='specify',
            closer='endspecify',
        )
        specify.addParseAction(self.add_specify)

        sub_module_declare = variable + variable + enclosedExpr(variable_list) + Suppress(';')
        sub_module_declare.addParseAction(self.add_submodule)

        gate_types = ['and', 'or', 'not', 'xor', 'xnor', 'nand', 'nor', 'buf']
        gate_types = [Keyword(gt) for gt in gate_types]
        gate_declare = reduce(lambda a, b: a | b, gate_types) + enclosedExpr(variable_list) + Suppress(';')
        gate_declare.setParseAction(self.add_gate)
        module_statement = variable_declare | gate_declare | sub_module_declare
        module_declare = enclosedExpr(
            variable + enclosedExpr(variable_list) + Suppress(";") + module_statement[...] + specify,
            opener="module",
            closer='endmodule',
        )
        module_declare.setParseAction(self.add_cell)
        return module_declare

    def get_grammar(self) -> ParserElement:
        variable_types = ['input', 'output', 'reg', 'wire']
        variable_types = [Keyword(vt) for vt in variable_types]
        variable_declare = reduce(lambda a, b: a | b, variable_types) + variable_list + Suppress(';')
        variable_declare.setParseAction(self.add_variable)

        primitive_declare = self.get_primitive_grammar(variable_declare)
        module_declare = self.get_module_grammar(variable_declare)
        module_declare = module_declare | ('module' + SkipTo('endmodule') + restOfLine)

        grammar = Group(Suppress(Keyword('`timescale')) + timescale + Suppress('/') + timescale) + \
            primitive_declare[...] + \
            Suppress(Keyword('`celldefine')) + module_declare[...] + Suppress(Keyword('`endcelldefine'))
        grammar.addParseAction(self.add_lib)

        grammar.ignore('//' + restOfLine)
        return grammar

    def add_table(self, s, loc, tok: List[ParseResults]):
        self.module_spec['table'] = [r.asList() for r in tok[1:]]

    def add_cell(self, s: str, loc: int, tok: List[ParseResults]):
        self.module_spec['type'] = tok[0]
        self.module_spec['name'] = tok[1]
        self.modules.append(self.module_spec)
        self.module_spec = {}

    def add_gate(self, s: str, loc: int, tok: List[ParseResults]):
        if 'gate' not in self.module_spec:
            self.module_spec['gate'] = []
        self.module_spec['gate'].append(tok.asList())

    def add_submodule(self, s: str, loc: int, tok: List[ParseResults]):
        if 'submodule' not in self.module_spec:
            self.module_spec['submodule'] = []
        self.module_spec['submodule'].append({
            'type': tok[0],
            'name': tok[1],
            'variables': tok[2].asList()
        })

    def add_specify(self, s: str, loc: int, tok: List[ParseResults]):
        self.module_spec['specify'] = tok.asList()[1:]

    def add_variable(self, s, loc, tok: List[ParseResults]):
        var_type = tok[0]
        if var_type not in self.module_spec:
            self.module_spec[var_type] = []
        self.module_spec[var_type].extend(tok[1])

    def add_lib(self, s, loc, tok: List[ParseResults]):
        self.timescale = tok[0].asList()
        self.timescale[0][0] = int(self.timescale[0][0])
        self.timescale[1][0] = int(self.timescale[1][0])

    def get_result(self):
        return {
            'timescale': self.timescale,
            'modules': self.modules,
        }

    def summary(self):
        print('Timescale: ', self.timescale)
        print('Modules:')
        pprint(self.modules)
