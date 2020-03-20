import os

from .vlib_parser import VLibParser


def read_vlib_file(path: str):
    if not os.path.isfile(path):
        raise ValueError(f"Vlib file {path} don't exist")

    parser = VLibParser()
    parser.grammar.parseFile(path)
    return parser.get_result()
