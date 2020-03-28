import os
import copy

from .vlib_parser import VLibParser


class VlibReader:

    @staticmethod
    def to_dict(l: list):
        d = {}
        for p in l:
            name = p.name
            del p['name']
            d[name] = p
        return d

    @staticmethod
    def filter_invalid(d: dict):
        """
        filter primitives/modules with sequential logic
        """
        filtered_d = {}
        for name, m in d.items():
            if 'declares' not in m:
                filtered_d[name] = m
                continue
            invalid = False
            for dec in m.declares:
                if dec.type == 'reg':
                    invalid = True
                    break
            if not invalid:
                filtered_d[name] = m

        return filtered_d

    @classmethod
    def organize_declares(cls, mods: dict):
        for k, m in mods.items():
            if 'declares' not in m:
                continue
            mods[k].declares = cls.organize_mod_declares(m.declares)
        return mods

    @staticmethod
    def organize_mod_declares(declares_list: list):
        variable_types = copy.deepcopy(VLibParser.variable_types)
        variable_types.remove('reg')
        declares_dict = {
            **{t: [] for t in variable_types},
            'gates': [],
            'submodules': [],
        }

        for d in declares_list:
            if 'type' in d:
                declare_type = d.type
                del d['type']
                declares_dict[declare_type].extend(d.ids)
            elif 'gate_type' in d:
                declares_dict['gates'].append(d)
            elif 'submodule_type' in d:
                declares_dict['submodules'].append(d)
        return declares_dict

    @classmethod
    def get_process_pipe(cls):
        def pipe(x):
            for f in [cls.to_dict, cls.filter_invalid, cls.organize_declares]:
                x = f(x)
            return x
        return pipe

    @classmethod
    def read_file(cls, path: str):
        if not os.path.isfile(path):
            raise ValueError(f"Vlib file {path} don't exist")

        parser = VLibParser()
        result = parser.get_grammar().parseFile(path)

        process_pipe = cls.get_process_pipe()
        result.primitives = process_pipe(result.primitives)
        result.modules = process_pipe(result.modules)
        return result
