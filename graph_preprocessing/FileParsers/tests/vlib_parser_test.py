from pprint import pprint

from ..vlib_parser import VLibParser


def test_parse_primitive():
    parser = VLibParser()
    test_case = """
        `timescale 1ps/1ps
        primitive udp_dff (q, d, clk, clr, set, notifier);
        output q;
        output r;
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
        `endcelldefine
        """.lstrip('\n')
    results = parser.get_grammar().parseString(test_case)

    pprint(dict(results))
    pprint(results.primitives[0].declares)
    pprint(results.primitives[0].declares[0])
    pprint(results.primitives[0].variables)

    assert len(results.primitives) == 1
    assert set(results.primitives[0].keys()) == {'name', 'declares', 'variables', 'table'}
    assert results.primitives[0].name == 'udp_dff'
    assert set(results.primitives[0].variables) == {'q', 'd', 'clk', 'clr', 'set', 'notifier'}
    assert results.primitives[0].declares[0].type == 'output'
    assert set(results.primitives[0].declares[0].ids) == {'q'}

    assert results.primitives[0].declares[-1].type == 'reg'
    assert set(results.primitives[0].declares[-1].ids) == {'q'}
    assert len(results.primitives[0].table) == 5


def test_parse_module():
    parser = VLibParser()
    test_case = """
        `timescale 1ps/1ps
        `celldefine
        module GEN_AND4_D2 (a1,a2,a3,a4,z);
          input a1;
          input a2;
          input a3;
          input a4;
          output z;
          and (z, a1, a2, a3, a4);
          specify
            (a1 => z)=(1, 1);
            (a2 => z)=(1, 1);
            (a3 => z)=(1, 1);
            (a4 => z)=(1, 1);
          endspecify
        endmodule
        `endcelldefine
        """.lstrip('\n')
    results = parser.get_grammar().parseString(test_case)

    pprint(results.modules[0].declares)
    assert set(results.modules[0].keys()) == {'name', 'declares', 'variables', 'specify'}
    assert results.modules[0].name == 'GEN_AND4_D2'
    assert len(results.modules[0].declares) == 6
    assert results.modules[0].declares[0].type == 'input'
    assert set(results.modules[0].declares[0].ids) == {'a1'}


def test_multiple_modules():
    parser = VLibParser()
    test_case = """
        `timescale 1ps/1ps
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
    results = parser.get_grammar().parseString(test_case)

    assert len(results.modules) == 2

    assert len(results.timescale) == 2
    assert tuple(results.timescale[0]) == ('1', 'ps')
    assert tuple(results.timescale[1]) == ('1', 'ps')


def test_ignore_comment():
    test_case = """
        `timescale 1ps/1ps
        primitive udp_dff (q, d, clk, clr, set, notifier);
        output q;
        input  d, clk, clr, set, notifier;
        reg    q;
          // i check_signal : o
        table
            // i0 i1 s :  z
            ?   ?   0 : 0 ;
        endtable
        endprimitive
        `celldefine
        `endcelldefine
        """.lstrip('\n')
    parser = VLibParser()
    results = parser.get_grammar().parseString(test_case)

    assert set(results.primitives[0].keys()) == {'name', 'declares', 'variables', 'table'}
    assert len(results.primitives[0].table) == 1


def test_parse_latch():
    test_case = """
        `timescale 1ps/1ps
        `celldefine
        module GEN_LATCH_D1 (d,e,q);
            input d;
            input e;
            output q;
            reg notifier;
            supply1 cdn;
            supply1 sdn;
            udp_tlat udpi0 (q, d, e, cdn, sdn, notifier);
            specify
                (d => q)=(1, 1);
                (posedge e => (q +: d))=(1, 1);
            endspecify
        endmodule
        `endcelldefine
        """.lstrip('\n')
    parser = VLibParser()
    results = parser.get_grammar().parseString(test_case)
    pprint(results.modules)

    assert len(results.primitives) == 0
    assert len(results.modules) == 0


def test_parse_submodule():
    test_case = """
        `timescale 1ps/1ps
        `celldefine
        module GEN_MUX2_D1 (i0,i1,s,z);
          input i0;
          input i1;
          input s;
          output z;
          udp_mux2 udpi0 (z, i0, i1, s);
          specify
            (i0 => z)=(1, 1);
            (i1 => z)=(1, 1);
            (posedge s => z) = (1, 1);
            (negedge s => z) = (1, 1);
          endspecify
        endmodule
        `endcelldefine
        """.lstrip('\n')
    parser = VLibParser()
    results = parser.get_grammar().parseString(test_case)

    assert set(results.modules[0].keys()) == {'name', 'declares', 'variables', 'specify'}
    assert set(results.modules[0].declares[-1].keys()) == {'submodule_type', 'id', 'variables'}
    assert results.modules[0].declares[-1].submodule_type == 'udp_mux2'
    assert results.modules[0].declares[-1].id == 'udpi0'
    assert set(results.modules[0].declares[-1].variables) == {'z', 'i0', 'i1', 's'}
