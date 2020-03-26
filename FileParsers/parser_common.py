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


def enclosedExpr(content=None, opener="(", closer=")", supress_font=False) -> ParserElement:
    if supress_font or opener == "(":
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

