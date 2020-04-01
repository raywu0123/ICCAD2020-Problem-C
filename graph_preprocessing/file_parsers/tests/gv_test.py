import os
from pprint import pprint

import pytest

from ..gv_parser import GVParser


@pytest.mark.parametrize(
    "test_case",
    [
        "input [31:0] in100_20x20;",
        "wire [3:0] wr;",
        "output [63:0] oo;",
    ],
)
def test_parse_multibit_declaration(test_case):
    grammar = GVParser.get_declaration()
    results = dict(grammar.parseString(test_case))
    pprint(results)
    assert set(results.keys()) == {'type', 'bitwidth', 'id'}


@pytest.mark.parametrize(
    "test_case,num_ids",
    [
        ("input in1, in2, in3;", 3),
        ("wire w1, w2;", 2),
        ("output o1;", 1),
    ],
)
def test_parse_variable_list_declaration(test_case, num_ids):
    grammar = GVParser.get_declaration()
    results = dict(grammar.parseString(test_case))
    pprint(results)

    assert set(results.keys()) == {'type', 'ids'}
    assert len(results['ids']) == num_ids


@pytest.mark.parametrize(
    "test_case,num_params,composite_num",
    [
        ("GENGEN123 g1(.a1(b1), .a2(b2));", 2, None),
        ("G g2( .x( y ) );", 1, None),
        ("""
         GEN_SYNC2C_D1 u_partition_m_reset_ICCADs_sync_reset_synced_rstn_ICCADs_NV_GENERIC_CELL ( 
        .clk(nvdla_core_clk), .clr_(
        u_partition_m_reset_ICCADs_sync_reset_synced_rstn_ICCADs_inreset_tm_), 
        .d(
        u_partition_m_reset_ICCADs_sync_reset_synced_rstn_ICCADs_inreset_xclamp_), .q(u_partition_m_reset_ICCADs_sync_reset_synced_rstn_ICCADs_reset_) );
        """, 4, None),
        ("""
        GEN_MUX2_D4 u_partition_m_reset_ICCADs_sync_reset_synced_rstn_ICCADs_UI_test_mode_inmux ( 
        .i0(dla_reset_rstn), .i1(direct_reset_), .s(1'b1), .z(
        u_partition_m_reset_ICCADs_sync_reset_synced_rstn_ICCADs_inreset_tm_)
         );
        """, 4, None),
        ("""
        GEN_XX gg(.dout({
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[14], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[13], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[12], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[11], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[10], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[9], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[8], 
        SYNOPSYS_UNCONNECTED__0, SYNOPSYS_UNCONNECTED__1, 
        SYNOPSYS_UNCONNECTED__2, SYNOPSYS_UNCONNECTED__3, 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[3], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[2], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[1], 
        u_partition_o_ICCADs_u_NV_NVDLA_cdp_ICCADs_u_dp_ICCADs_sync2ocvt_pd[0]})
        );
        """, 1, 15)
    ],
)
def test_parse_cell(test_case, num_params, composite_num):
    grammar = GVParser.get_cell()
    results = grammar.parseString(test_case)
    pprint(dict(results))
    assert set(results.keys()) == {'cell_type', 'id', 'parameters'}
    assert len(results.parameters) == num_params

    if composite_num is not None:
        assert len(results.parameters[0]) == composite_num + 1


@pytest.mark.parametrize(
    "test_case,num_params,num_declares,num_cells",
    [
        ("""
        module myMod (a, b, c, d);
            input i1, i2;
            output [10:0] o;
            wire w1, w2;
            SUB submod1(.x(w1), .y(w2), .z(o)
            );
            SUB submod2(.xx(ww1[5]), .yy(ww2), .zz(oo));
        endmodule
        """, 4, 3, 2),
    ],
)
def test_parse_module(test_case, num_params, num_declares, num_cells):
    grammar = GVParser.get_module()
    results = grammar.parseString(test_case.strip('\n'))

    assert {'id', 'parameters', 'body'}.difference(set(results.keys())) == set()
    assert len(results['parameters']) == num_params
    assert len(results['body']) == num_declares + num_cells
    for item in results['body']:
        assert ('id' in item.keys() or 'ids' in item.keys())
        if 'type' not in item.keys():
            assert set(item.keys()) == {'id', 'parameters', 'cell_type'}


@pytest.mark.parametrize(
    "test_case",
    [
        "assign csb2cmac_a_req_prdy = 1'b1;",
        "assign csb2cmac_a_req_prdy = 1'b0;",
    ],
)
def test_parse_assign(test_case):
    grammar = GVParser.get_assign()
    results = dict(grammar.parseString(test_case.strip('\n')))
    pprint(results)
    assert len(list(results)) == 3


def test_read_file():
    path = os.path.join('./test-cases/NV_NVDLA_partition_m_dc_24x33x55_5x5x55x25_int8/NV_NVDLA_partition_m_GEN.gv')
    results = GVParser.read_file(path)

    pprint(results.keys())
    assert {'id', 'parameters', 'io', 'assign', 'cells'}.difference(set(results.keys())) == set()

    body_io_count, body_assign_count, body_cell_count = 0, 0, 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if (
                    line.startswith('input') or
                    line.startswith('output') or
                    line.startswith('wire')
            ):
                body_io_count += 1
            elif line.startswith('assign'):
                body_assign_count += 1
            elif line.startswith('GEN_'):
                body_cell_count += 1

    assert len(results.io) == body_io_count
    assert len(results.assign) == body_assign_count
    assert len(results.cells) == body_cell_count
