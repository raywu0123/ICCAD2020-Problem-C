from itertools import chain
from typing import List, Set, Hashable


class Graph:

    def __init__(self):
        self.in_nodes_of_key = {}
        self.out_nodes_of_key = {}

    def add_node(self, idd: Hashable):
        if idd in self.in_nodes_of_key.keys() or idd in self.out_nodes_of_key.keys():
            raise ValueError(f'Node id {idd} already used.')
        else:
            self.in_nodes_of_key[idd] = []
            self.out_nodes_of_key[idd] = []

    def add_edge(self, from_node_id: Hashable, to_node_id: Hashable):
        self.out_nodes_of_key[from_node_id].append(to_node_id)
        self.in_nodes_of_key[to_node_id].append(from_node_id)

    def get_initial_nodes(self):
        return {n for n, l in self.in_nodes_of_key.items() if len(l) == 0}

    def get_schedule_layers(self) -> List[Set[Hashable]]:
        L = [self.get_initial_nodes()]
        while True:
            next_layer = set()
            for n in L[-1]:
                for m in self.out_nodes_of_key[n]:
                    self.in_nodes_of_key[m].remove(n)
                    if len(self.in_nodes_of_key[m]) == 0:
                        next_layer.add(m)
                self.out_nodes_of_key[n].clear()

            if len(next_layer) == 0:
                break
            L.append(next_layer)

        for l in chain(self.in_nodes_of_key.values(), self.out_nodes_of_key.values()):
            if not len(l) == 0:
                raise RuntimeError('Graph contains cycle')
        return L
