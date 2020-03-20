import os
from tempfile import NamedTemporaryFile
from pprint import pprint

from ..vlib_reader import read_vlib_file


def test_reader_custom_small():
    f = NamedTemporaryFile(mode='w')
    test_case = """
        `timescale 1ps/1ps
        primitive udp_dff (q, d, clk, clr, set, notifier);
        output q;
        input  d, clk, clr, set, notifier;
        reg    q;
        table
            ?   ?   0   ?   ? : ? : 0 ;
            ?   ?   1   x   ? : 0 : x ;
            0 (01)  ?   1   ? : ? : 0 ;
            0   *   ?   1   ? : 0 : 0 ;
            ? (1?)  1   1   ? : ? : - ;
        endtable
        endprimitive
        `celldefine
        module GEN_XNOR2_D1 (a1,a2,zn);
          input a1;
          input a2;
          output zn;
          xnor (zn, a1, a2);
          specify
            (posedge a1 => zn) = (1, 1);
            (negedge a1 => zn) = (1, 1);
            (posedge a2 => zn) = (1, 1);
            (negedge a2 => zn) = (1, 1);
          endspecify
        endmodule
        module GEN_XNOR2_D2 (a1,a2,zn);
          input a1;
          input a2;
          output zn;
          xnor (zn, a1, a2);
          specify
            (posedge a1 => zn) = (1, 1);
            (negedge a1 => zn) = (1, 1);
            (posedge a2 => zn) = (1, 1);
            (negedge a2 => zn) = (1, 1);
          endspecify
        endmodule
        `endcelldefine
        """.lstrip('\n')
    print(
        test_case,
        file=f,
        flush=True,
    )
    result = read_vlib_file(path=f.name)
    pprint(result)


def test_reader():
    path = os.path.join('./test-cases/GENERIC_STD_CELL.vlib')
    result = read_vlib_file(path)
    with open(path, 'r') as f:
        s = f.read()

    s = s.replace("endmodule", "")
    s = s.replace('module', '#')
    s = s.replace("endprimitive", '')
    s = s.replace("primitive", '#')
    splits = s.split("#")[1:]

    valid_modules = [
        m.split('(')[0].strip() for m in splits
        if 'always' not in m and '+:' not in m
    ]

    parsed_modules = {m['name'] for m in result['modules']}
    valid_modules_not_parsed = [
        m for m in valid_modules if m not in parsed_modules
    ]
    assert len(valid_modules_not_parsed) == 0
    assert set(result.keys()) == {'timescale', 'modules'}
