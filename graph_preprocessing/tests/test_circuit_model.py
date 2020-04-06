from collections import namedtuple
from ..circuit_model import Circuit, SINGLE_BIT_INDEX


class SchemaInstance:

    def __init__(self, namedtuple_instance):
        self.tuple = namedtuple_instance

    def __getattr__(self, item):
        return getattr(self.tuple, item)

    def __contains__(self, item):
        return item in self.tuple._fields


class Schema:

    def __init__(self, typename: str, fieldnames):
        self.tuple = namedtuple(typename, fieldnames)

    def __call__(self, *args, **kwargs):
        return SchemaInstance(self.tuple(*args, **kwargs))


CellSpec = Schema('cell_spec', ['declares'])
StdInfo = Schema('std_info', ['primitives', 'modules'])
GVInfo = Schema('gv_info', ['io', 'assign', 'cells'])

SingleIO = Schema('io', ['ids', 'type'])
BusIO = Schema('io', ['id', 'bitwidth', 'type'])
Cell = Schema('cell', ['cell_type', 'id', 'parameters'])


mock_std_info = StdInfo(
    primitives={
        'and': CellSpec(declares={'input': ['a', 'b'], 'output': ['z']})
    },
    modules={
    }
)

def test_circuit_parallel():
    mock_gv_info_parallel = GVInfo(
        io=[
            BusIO(id='ibus_1', bitwidth=[0, 3], type='input'),
            SingleIO(ids=['o1', 'o2', 'o3'], type='output'),
        ],
        assign=[],
        cells=[
            Cell(cell_type='and', id='and1', parameters=["z o1", "b ibus_1[0]", "a ibus_1[1]"]),
            Cell(cell_type='and', id='and2', parameters=["z o2", "b ibus_1[2]", "a ibus_1[3]"]),
            Cell(cell_type='and', id='and3', parameters=["a o2", "b o1", "z o3"]),
        ]
    )
    circuit = Circuit(gv_info=mock_gv_info_parallel, std_cell_info=mock_std_info)
    circuit.summary()

    assert set(circuit.wire_inputs.keys()) == {
        ("1'b0", SINGLE_BIT_INDEX),
        ("1'b1", SINGLE_BIT_INDEX),
        ("ibus_1", 0), ("ibus_1", 1), ("ibus_1", 2), ("ibus_1", 3),
        ("o1", SINGLE_BIT_INDEX),
        ("o2", SINGLE_BIT_INDEX),
        ("o3", SINGLE_BIT_INDEX),
    }
    assert len(circuit.schedule_layers) == 2
    assert len(circuit.mem_alloc_schedule) == 2
    assert len(circuit.mem_free_schedule) == 2
    assert circuit.mem_free_schedule[0] == {("ibus_1", 0), ("ibus_1", 1), ("ibus_1", 2), ("ibus_1", 3)}
    assert circuit.mem_free_schedule[1] == {("o1", SINGLE_BIT_INDEX), ("o2", SINGLE_BIT_INDEX), ("o3", SINGLE_BIT_INDEX)}


def test_circuit_sequential():
    mock_gv_info_sequential = GVInfo(
        io=[
            SingleIO(ids=['i1'], type='input'),
            BusIO(id='ibus1', bitwidth=[3, 2], type='input'),
            BusIO(id='o1', bitwidth=(1, 0), type='output'),
            SingleIO(ids=['o2'], type='output'),
            SingleIO(ids=['w'], type='wire'),
        ],
        assign=[['o1[0]', "1'b1"]],
        cells=[
            Cell(cell_type='and', id='and1', parameters=["a ibus1[2]", "b i1", "z w"]),
            Cell(cell_type='and', id='and2', parameters=["a w", "b ibus1[3]", "z o1[1]"]),
            Cell(cell_type='and', id='and3', parameters=["a o1[0]", "b o1[1]", "z o2"]),
        ],
    )
    circuit = Circuit(gv_info=mock_gv_info_sequential, std_cell_info=mock_std_info)
    circuit.summary()

    assert set(circuit.wire_inputs.keys()) == {
        ("1'b0", SINGLE_BIT_INDEX),
        ("1'b1", SINGLE_BIT_INDEX),
        ("i1", SINGLE_BIT_INDEX),
        ("o2", SINGLE_BIT_INDEX),
        ("w", SINGLE_BIT_INDEX),
        ("ibus1", 2),
        ("ibus1", 3),
        ("o1", 1),
        # ("o1", 0),  assigned
    }
    assert len(circuit.schedule_layers) == 3
    assert len(circuit.mem_alloc_schedule) == 3
    assert len(circuit.mem_free_schedule) == 3
    assert circuit.mem_free_schedule[0] == {("ibus1", 2), ("i1", SINGLE_BIT_INDEX)}
    assert circuit.mem_free_schedule[1] == {("ibus1", 3), ("w", SINGLE_BIT_INDEX)}
    assert circuit.mem_free_schedule[2] == {("o1", 0), ("o1", 1), ("o2", SINGLE_BIT_INDEX)}


def test_circuit_reuse():
    mock_gv_info_reuse = GVInfo(
        io=[
            SingleIO(ids=['i1', 'i2'], type='input'),
            SingleIO(ids=['o'], type='output'),
            SingleIO(ids=['w'], type='wire'),
        ],
        assign=[],
        cells=[
            Cell(cell_type='and', id='and1', parameters=["a i1", "b i2", "z w"]),
            Cell(cell_type='and', id='and2', parameters=["a w", "b i2", "z o"]),
        ],
    )
    circuit = Circuit(gv_info=mock_gv_info_reuse, std_cell_info=mock_std_info)
    circuit.summary()

    assert set(circuit.wire_inputs.keys()) == {
        ("1'b0", SINGLE_BIT_INDEX),
        ("1'b1", SINGLE_BIT_INDEX),
        ("i1", SINGLE_BIT_INDEX),
        ("i2", SINGLE_BIT_INDEX),
        ("w", SINGLE_BIT_INDEX),
        ("o", SINGLE_BIT_INDEX),
    }
    assert len(circuit.schedule_layers) == 2
    assert len(circuit.mem_alloc_schedule) == 2
    assert len(circuit.mem_free_schedule) == 2
    assert circuit.mem_free_schedule[0] == {("i1", SINGLE_BIT_INDEX)}
    assert circuit.mem_free_schedule[1] == {("i2", SINGLE_BIT_INDEX), ("w", SINGLE_BIT_INDEX), ("o", SINGLE_BIT_INDEX)}
