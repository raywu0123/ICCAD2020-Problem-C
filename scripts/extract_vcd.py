from sys import argv

from .VCDReader import VCDReader


if __name__ == '__main__':
    vcd_path = argv[1]
    vcd_reader = VCDReader(vcd_path, wire_names=set(argv[3:]))

    real_blocks = [b for b in vcd_reader.blocks if len(b[1]) > 0]
    
    out_path = argv[2]
    fout = open(out_path, 'w')

    print("Writing result")
    fout.write(vcd_reader.header)
    for b in real_blocks:
        print(f'#{b[0]}', file=fout)
        for line in b[1]:
            fout.write(line)
