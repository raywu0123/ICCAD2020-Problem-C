from functools import reduce

from pyparsing import (
    Optional,
    Word,
    Keyword,
    Group,
    Suppress,
    delimitedList,
    alphas,
    alphanums,
    ParserElement,
    Combine,
    Literal,
    nums,
)


def enclosedExpr(content=None, opener="(", closer=")", supress_font=False) -> ParserElement:
    if supress_font or opener == "(":
        opener = Suppress(opener)
    expr = opener + content + Suppress(closer)
    return expr


posnegedge_keyword = Optional(Keyword('posedge') | Keyword('negedge'))

variable = Word(alphas + '_', alphanums + "_")
variable_list = Group(delimitedList(variable))

floatnum = Combine(Word(nums) + '.' + Word(nums))

timeunits = ['ps', 'ns']
timeunits = [Literal(tu) for tu in timeunits]
timescale = Group(Word(nums) + reduce(lambda a, b: a | b, timeunits))

