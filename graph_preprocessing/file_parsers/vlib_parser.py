from functools import reduce

from pyparsing import (
    Keyword,
    Word,
    Group,
    Suppress,
    delimitedList,
    nums,
    ParserElement,
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

    variable_types = ['input', 'output', 'reg', 'wire', 'supply1', 'supply0']
    gate_types = ['and', 'or', 'not', 'xor', 'xnor', 'nand', 'nor', 'buf']

    @classmethod
    def get_primitive_grammar(cls):
        basic_table_token = Word("01?x-*", max=1)
        bracket_table_token = Group("(" + Word("01?x-", max=2) + ")")
        table_token = basic_table_token | bracket_table_token
        table_line = Group(delimitedList(Group(table_token[1, ...]), delim=":")) + Suppress(';\n')
        table_declare = enclosedExpr(
            content=table_line[1, ...],
            opener='table',
            closer='endtable',
            supress_front=True
        )
        primitive_declare = enclosedExpr(
            variable('name') + enclosedExpr(variable_list('variables')) + Suppress(";") +
            Group(cls.variable_declare())[...]('declares') + table_declare('table'),
            opener="primitive",
            closer='endprimitive',
            supress_front=True,
        )
        return primitive_declare

    @classmethod
    def get_module_grammar(cls):
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

        sub_module_declare = variable('submodule_type') + variable('id') + enclosedExpr(variable_list('variables')) + Suppress(';')

        gate_types = [Keyword(gt) for gt in cls.gate_types]
        gate_declare = reduce(lambda a, b: a | b, gate_types)('gate_type') + enclosedExpr(variable_list('variables')) + Suppress(';')
        module_statement = Group(cls.variable_declare() | gate_declare | sub_module_declare)
        module_declare = enclosedExpr(
            variable('name') + enclosedExpr(variable_list('variables')) + Suppress(";") +
            module_statement[...]('declares') + specify('specify'),
            opener="module",
            closer='endmodule',
            supress_front=True,
        )
        return module_declare

    @classmethod
    def variable_declare(cls):
        variable_types = [Keyword(vt)('type') for vt in cls.variable_types]
        variable_declare = reduce(lambda a, b: a | b, variable_types) + variable_list('ids') + Suppress(';')
        return variable_declare

    def get_grammar(self) -> ParserElement:
        primitive_declare = Group(self.get_primitive_grammar())
        module_declare = Group(self.get_module_grammar())
        module_declare = module_declare | Suppress('module' + SkipTo('endmodule') + restOfLine)

        grammar = Group(Suppress(Keyword('`timescale')) + timescale + Suppress('/') + timescale)('timescale') + \
            primitive_declare[...]('primitives') + \
            Suppress(Keyword('`celldefine')) + module_declare[...]('modules') + Suppress(Keyword('`endcelldefine'))

        grammar.ignore('//' + restOfLine)
        return grammar
