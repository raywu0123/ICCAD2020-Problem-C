import pytest
import random

from ..graph import Graph


def generate_random_DAG(n_nodes: int):
    edges = []
    for n in range(1, n_nodes + 1):
        for other_node in range(1, n):
            if random.random() > 0.5:
                edges.append((other_node, n))
    return n_nodes, edges


@pytest.mark.parametrize(
    "n_nodes, edges", [
        (5, [(2, 1), (3, 1), (4, 2), (5, 2)]),
        (generate_random_DAG(5)),
        (generate_random_DAG(10)),
        (generate_random_DAG(100)),
    ]
)
def test_get_schedule_layers(n_nodes, edges):
    graph = Graph()
    for n in range(1, n_nodes + 1):
        graph.add_node(n)

    for from_node, to_node in edges:
        graph.add_edge(from_node, to_node)

    try:
        schedule_layers = graph.get_schedule_layers()
        node_to_schedule_index = {
            n: i
            for i, node_set in enumerate(schedule_layers)
            for n in node_set
        }
        for from_node, to_node in edges:
            assert node_to_schedule_index[from_node] < node_to_schedule_index[to_node]
    except RuntimeError as err:
        print(err)
        print(edges)
        assert False
