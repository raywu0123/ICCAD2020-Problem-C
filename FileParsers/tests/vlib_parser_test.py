from ..vlib_parser import VLibParser


def test_parse_primitive():
    parser = VLibParser()
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
        `endcelldefine
        """.lstrip('\n')
    parser.grammar.parseString(test_case)
    parser.summary()

    assert set(parser.modules[0].keys()) == {'name', 'type', 'reg', 'input', 'output', 'table'}
    assert parser.modules[0]['type'] == 'primitive'
    assert parser.modules[0]['name'] == 'udp_dff'
    assert set(parser.modules[0]['input']) == {'d', 'clk', 'clr', 'set', 'notifier'}
    assert set(parser.modules[0]['output']) == {'q'}
    assert set(parser.modules[0]['reg']) == {'q'}
    assert len(parser.modules[0]['table']) == 5


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
    parser.grammar.parseString(test_case)
    parser.summary()

    assert(set(parser.modules[0].keys()) == {'name', 'type', 'gate', 'specify', 'input', 'output'})


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
    parser.grammar.parseString(test_case)
    parser.summary()
    assert len(parser.modules) == 2
    assert parser.timescale == [[1, 'ps'], [1, 'ps']]


def test_ignore_comment():
    parser = VLibParser()
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
    parser.grammar.parseString(test_case)
    parser.summary()

    assert(set(parser.modules[0].keys()) == {'name', 'type', 'reg', 'input', 'output', 'table'})
    assert(len(parser.modules[0]['table']) == 1)


def test_parse_latch():
    parser = VLibParser()
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
    parser.grammar.parseString(test_case)
    parser.summary()
    assert len(parser.modules) == 0


def test_parse_submodule():
    parser = VLibParser()
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
    parser.grammar.parseString(test_case)
    parser.summary()
    assert set(parser.modules[0].keys()) == {'input', 'output', 'submodule', 'specify', 'name', 'type'}
