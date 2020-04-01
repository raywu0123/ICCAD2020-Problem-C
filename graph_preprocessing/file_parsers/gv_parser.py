import os

from pyparsing import (
    Group,
    Combine,
    Suppress,
    ParserElement,
    Optional,
    delimitedList,
    restOfLine,
    pyparsing_common,
)

from .parser_common import (
    enclosedExpr,
    variable,
    variable_list,
    make_keyword,
    bitwidth,
    bits,
)


class GVParser:

    @staticmethod
    def get_declaration() -> ParserElement:
        type_keyword = make_keyword('wire') | make_keyword('input') | make_keyword('output')
        multibit_variable = bitwidth('bitwidth') + variable('id')
        declaration = type_keyword('type') + (multibit_variable | variable_list('ids')) + Suppress(";")
        return declaration

    @staticmethod
    def get_cell() -> ParserElement:
        custom_bitwidth = '[' + pyparsing_common.integer + Optional(':' + pyparsing_common.integer) + ']'
        wire = Combine(
            variable + Optional(custom_bitwidth)
        )

        composite = Suppress(Optional('{')) + delimitedList(wire | Combine(bits)) + Suppress(Optional('}'))
        assignment = Group(
            Suppress('.') + variable + enclosedExpr(composite)
        )
        parameters = delimitedList(assignment)
        cell = variable('cell_type') + variable('id') + enclosedExpr(parameters)('parameters') + Suppress(';')
        return cell

    @classmethod
    def get_assign(cls):
        custom_bitwidth = '[' + pyparsing_common.integer + Optional(':' + pyparsing_common.integer) + ']'
        return Suppress(make_keyword('assign')) + Combine(variable + Optional(custom_bitwidth))('lhs') + Suppress("=") \
               + Combine(bits) + Suppress(';')

    @classmethod
    def get_module(cls) -> ParserElement:
        line = Group(cls.get_cell())('cell') | Group(cls.get_declaration())('declare') | Group(cls.get_assign())('assign')
        grammar = make_keyword('module') \
            + variable('id') + enclosedExpr(variable_list('parameters')) + Suppress(';') \
            + line[...]('body') + make_keyword('endmodule')
        grammar.ignore('//' + restOfLine)
        return grammar

    @classmethod
    def read_file(cls, path: str):
        if not os.path.isfile(path):
            raise ValueError(f"SDF file {path} don't exist")

        grammar = cls.get_module()
        results = grammar.parseFile(path)
        results['io'], results['assign'], results['cells'] = [], [], []

        for line in results.body:
            if 'cell_type' in line:
                args = [' '.join(p) for p in line.parameters]
                line.parameters = args

        for line in results.body:
            if 'type' in line:
                bucket_key = 'io'
            elif 'value' in line:
                bucket_key = 'assign'
            elif 'cell_type' in line:
                bucket_key = 'cells'
            else:
                raise ValueError(f'Unrecognized line type {line}')
            results[bucket_key].append(line)

        del results['body']
        return results
