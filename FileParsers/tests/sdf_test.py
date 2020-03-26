import os
from pprint import pprint

from ..sdf import SDFParser


def test_parse_cell():
    test_case = """
    (CELL
        (CELLTYPE "NV_NVDLA_partition_m")
        (INSTANCE U21629)
        (DELAY
            (ABSOLUTE
                (iopath a1 z (0.014:0.014:0.014) (0.013:0.013:0.013))
                (iopath (negedge ci) s (0.024:0.024:0.024) (0.019:0.019:0.019))
            )
        )
    )
    """.lstrip('\n')
    parser = SDFParser()
    cell = parser.get_cell_grammar()
    result = cell.parseString(test_case)
    assert len(result) > 0


def test_parse_first_cell():
    test_case = """
    (CELL
        (CELLTYPE "NV_NVDLA_partition_m")
        (INSTANCE)
        (DELAY
            (ABSOLUTE
            (INTERCONNECT u_NV_NVDLA_cmac_ICCADs_u_core_ICCADs_u_mac_0_ICCADs_sum_out_d0_d1_reg_17_/q U21631/a1 (0.000:0.000:0.000))
            )
        )
    )
    """.lstrip('\n')
    parser = SDFParser()
    cell = parser.get_cell_grammar()
    result = dict(cell.parseString(test_case))
    pprint(result)
    assert set(result.keys()) == {'type', 'delay'}


def test_read_header():
    path = os.path.join('./test-cases/NV_NVDLA_partition_m_dc_24x33x55_5x5x55x25_int8/NV_NVDLA_partition_m_GEN.sdf')
    with open(path, 'r') as f:
        header, _ = SDFParser.read_header(f)

    assert set(header.keys()) == {
        'SDFVERSION',
        'DESIGN',
        "DATE",
        "VENDOR",
        "PROGRAM",
        "VERSION",
        "DIVIDER",
        "VOLTAGE",
        "PROCESS",
        "TEMPERATURE",
        "TIMESCALE",
    }


def test_read_file():
    path = os.path.join('./test-cases/NV_NVDLA_partition_m_dc_24x33x55_5x5x55x25_int8/NV_NVDLA_partition_m_GEN.sdf')
    parser = SDFParser()
    header, cells = parser.read_file(path)

    with open(path, 'r') as f:
        whole_file = f.read()
    num_cells = whole_file.count("(CELLTYPE")
    assert len(cells) == num_cells

    for cell in cells:
        assert isinstance(cell, dict)
        assert 'type' in cell
        assert 'delay' in cell
