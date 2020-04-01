from functools import reduce

from pyparsing import (
    Optional,
    Word,
    Keyword,
    Group,
    Suppress,
    delimitedList,
    ParserElement,
    Literal,
    nums,
    pyparsing_common,
)


def enclosedExpr(content=None, opener="(", closer=")", supress_front=False) -> ParserElement:
    if supress_front or opener == "(":
        opener = Suppress(opener)
    expr = opener + content + Suppress(closer)
    return expr


def make_keyword(w: str):
    return Keyword(w)


posnegedge_keyword = Optional(Keyword('posedge') | Keyword('negedge'))

variable = pyparsing_common.identifier
variable_list = Group(delimitedList(variable))

timeunits = ['ps', 'ns']
timeunits = [Literal(tu) for tu in timeunits]
timescale = Group(Word(nums) + reduce(lambda a, b: a | b, timeunits))

bitwidth = Group(enclosedExpr(
    pyparsing_common.integer + Optional(Suppress(":") + pyparsing_common.integer),
    opener='[',
    closer=']',
    supress_front=True,
    ))

bits = pyparsing_common.integer('n_bits') + "'b" + pyparsing_common.integer('value')
