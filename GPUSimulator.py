from sys import argv
import pickle

from gpu_simulator.vcd_reader import VCD
from gpu_simulator.result_buffer import ResultBuffer


if __name__ == '__main__':
    if len(argv) != 5:
        raise ValueError(
            "Usage: GPUSimulator.cu.py "
            "<intermediate_representation.file> "
            "<input.vcd> "
            "<SAIF_or_VCD_flag> " 
            "[SAIF_or_output_VCD.saif.vcd]"
        )
    with open(argv[1], 'rb') as f_in:
        intermediate_info = pickle.load(f_in)

    design_name = intermediate_info['design_name']
    circuit = intermediate_info['circuit']

    print(f'Design name: {design_name}')

    input_vcd = VCD()
    input_vcd.read(argv[2])
    input_vcd.summary()
    circuit.finalize(input_vcd.get_wire_keys())
    exit()
    result_buffer = ResultBuffer()
    for batch in input_vcd:
        circuit.set_input(batch)
        batch_output = circuit.forward()
        result_buffer.update(batch_output)

    if argv[3] == 'SAIF':
        result_buffer.write_saif(argv[4])
    elif argv[3] == 'VCD':
        result_buffer.write_vcd(argv[4])
    else:
        raise ValueError("Third argument should be SAIF or VCD")
