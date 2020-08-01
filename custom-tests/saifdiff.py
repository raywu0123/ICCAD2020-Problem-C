# usage: saifdiff.py <saif_file_1> <saif_file_2>

import sys
import re
from pprint import pprint


def read_saif(file_path):
    f = open(file_path, 'r')
    f_string = f.read()
    TIMESCALESTRING = re.search(r'TIMESCALE\s+([0-9]+)\s+([pnuf]s)', f_string)
    TIMEVALUE = TIMESCALESTRING.group(1)
    TIMEUNIT = TIMESCALESTRING.group(2)
    if 'fs' in TIMEUNIT:
        TIMESCALE = 0.001
    elif 'ps' in TIMEUNIT:
        TIMESCALE = 1
    elif 'ns' in TIMEUNIT:
        TIMESCALE = 1000
    elif 'us' in TIMEUNIT:
        TIMESCALE = 1000000
    else:
        print('not a valid TIMESCALE in ' + file_path + ' !!! Error! Exiting...')
        sys.exit(1)

    TIMESCALE *= int(TIMEVALUE)
    DURATIONSTRING = re.search(r'DURATION\s+([0-9]+)', f_string)
    DURATION = TIMESCALE * int(DURATIONSTRING.group(1))

    NETS = re.findall(r'\s+\(\s*([_a-zA-Z\\\[\]0-9]+)\s*((\(\s*[TI][01XZCG]\s+[0-9]+\s*\)\s*){3,6})\)', f_string)
    d = {}
    for i in range(len(NETS)):
        T0 = int(re.search('T0\s+(\d+)', NETS[i][1]).group(1)) * TIMESCALE
        T1 = int(re.search('T1\s+(\d+)', NETS[i][1]).group(1)) * TIMESCALE
        TX = int(re.search('TX\s+(\d+)', NETS[i][1]).group(1)) * TIMESCALE
        if 'TZ' in NETS[i][1]:
            TZ = int(re.search('TZ\s+(\d+)', NETS[i][1]).group(1)) * TIMESCALE
        else:
            TZ = 0
        TX += TZ
        d[NETS[i][0]] = [T0, T1, TX]
        if sum(d[NETS[i][0]]) != DURATION:
            print(f'NET1 values for {NETS[i][0]} don\'t sum to duration')

    return {
        "DURATION": DURATION,
        "signals": d,
    }


golden_saif = sys.argv[1]
compare_saif = sys.argv[2]

golden = read_saif(golden_saif)
compare = read_saif(compare_saif)

# first, compare duration and get timescale of 2 saifs

if golden['DURATION'] != compare['DURATION']:
    print('SAIF durations don\'t match!!! Error! Exiting...')
    sys.exit(1)

extra_1 = set(golden['signals'].keys()).difference(compare['signals'].keys())
extra_2 = set(compare['signals'].keys()).difference(golden['signals'].keys())

if len(extra_1):
    print('In file1 but not in file2:')
    pprint(extra_1, )

if len(extra_2):
    print('In file2 but not in file1:')
    pprint(extra_2)

common_keys = set(golden['signals'].keys()).intersection(set(compare['signals'].keys()))
golden_common_signals = {k: v for k, v in golden['signals'].items() if k in common_keys}
compare_common_signals = {k: v for k, v in compare['signals'].items() if k in common_keys}

if golden_common_signals == compare_common_signals:
    print("SAIFs match! No Errors! Congratulations!")
    sys.exit(0)

cnt = 0
for key in common_keys:
    if golden['signals'][key] != compare['signals'][key]:
        print('signal values for ' + str(key) + ' is not the same!')
    else:
        cnt += 1

print(f'Passed {cnt / len(common_keys):.0%} ({cnt}/{len(common_keys)})')
