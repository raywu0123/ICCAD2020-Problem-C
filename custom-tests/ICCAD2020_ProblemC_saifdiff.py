# usage: python3 ICCAD2020_ProblemC_saifdiff.py <saif_file_1> <saif_file_2>

import sys
import re

golden_saif = sys.argv[1]
compare_saif = sys.argv[2]

f1 = open(golden_saif, 'r')
f2 = open(compare_saif, 'r')
f1_string = f1.read()
f2_string = f2.read()

# first, compare duration and get timescale of 2 saifs
TIMESCALESTRING_1 = re.search(r'TIMESCALE\s+([0-9]+)\s+([pnuf]s)', f1_string)
# normalize everything to ps
TIMEVALUE_1 = TIMESCALESTRING_1.group(1)
TIMEUNIT_1 = TIMESCALESTRING_1.group(2)
if 'fs' in TIMEUNIT_1:
    TIMESCALE_1 = 0.001
elif 'ps' in TIMEUNIT_1:
    TIMESCALE_1 = 1
elif 'ns' in TIMEUNIT_1:
    TIMESCALE_1 = 1000
elif 'us' in TIMEUNIT_1:
    TIMESCALE_1 = 1000000
else:
    print('not a valid TIMESCALE in ' + golden_saif + ' !!! Error! Exiting...')
    sys.exit(1)

TIMESCALE_1 *= int(TIMEVALUE_1)
DURATIONSTRING_1 = re.search(r'DURATION\s+([0-9]+)', f1_string)
DURATION_1 = TIMESCALE_1 * int(DURATIONSTRING_1.group(1))

TIMESCALESTRING_2 = re.search(r'TIMESCALE\s+([0-9]+)\s+([pnuf]s)', f2_string)
# normalize everything to ps
TIMEVALUE_2 = TIMESCALESTRING_2.group(1)
TIMEUNIT_2 = TIMESCALESTRING_2.group(2)
if 'fs' in TIMEUNIT_2:
    TIMESCALE_2 = 0.001
elif 'ps' in TIMEUNIT_2:
    TIMESCALE_2 = 1
elif 'ns' in TIMEUNIT_2:
    TIMESCALE_2 = 1000
elif 'us' in TIMEUNIT_2:
    TIMESCALE_2 = 1000000
else:
    print('not a valid TIMESCALE in ' + compare_saif + ' !!! Error! Exiting...')
    sys.exit(1)

TIMESCALE_2 *= int(TIMEVALUE_2)
DURATIONSTRING_2 = re.search(r'DURATION\s+([0-9]+)', f2_string)
DURATION_2 = TIMESCALE_2 * int(DURATIONSTRING_2.group(1))

if DURATION_1 != DURATION_2:
    print('SAIF durations don\'t match!!! Error! Exiting...')
    sys.exit(1)

dict_1 = {}
dict_2 = {}

NETS_1 = re.findall(r'\s+\(\s*([_a-zA-Z\\\[\]0-9]+)\s*((\(\s*[TI][01XZCG]\s+[0-9]+\s*\)\s*){3,6})\)', f1_string)

for i in range(len(NETS_1)):
    T0 = int(re.search('T0\s+(\d+)', NETS_1[i][1]).group(1)) * TIMESCALE_1
    T1 = int(re.search('T1\s+(\d+)', NETS_1[i][1]).group(1)) * TIMESCALE_1
    TX = int(re.search('TX\s+(\d+)', NETS_1[i][1]).group(1)) * TIMESCALE_1
    if 'TZ' in NETS_1[i][1]:
        TZ = int(re.search('TZ\s+(\d+)', NETS_1[i][1]).group(1)) * TIMESCALE_1
    else:
        TZ = 0
    dict_1[NETS_1[i][0]] = [T0, T1, TX, TZ]

NETS_2 = re.findall(r'\s+\(\s*([_a-zA-Z\\\[\]0-9]+)\s*((\(\s*[TI][01XZCG]\s+[0-9]+\s*\)\s*){3,6})\)', f2_string)

for i in range(len(NETS_2)):
    T0 = int(re.search('T0\s+(\d+)', NETS_2[i][1]).group(1)) * TIMESCALE_2
    T1 = int(re.search('T1\s+(\d+)', NETS_2[i][1]).group(1)) * TIMESCALE_2
    TX = int(re.search('TX\s+(\d+)', NETS_2[i][1]).group(1)) * TIMESCALE_2
    if 'TZ' in NETS_2[i][1]:
        TZ = int(re.search('TZ\s+(\d+)', NETS_2[i][1]).group(1)) * TIMESCALE_2
    else:
        TZ = 0
    dict_2[NETS_2[i][0]] = [T0, T1, TX, TZ]

if dict_1 == dict_2:
    print("SAIFs match! No Errors! Congratulations!")
    sys.exit(0)

extra_1 = (list(set(dict_1.keys()).difference(dict_2.keys())))
extra_2 = (list(set(dict_2.keys()).difference(dict_1.keys())))

if len(extra_1):
    print(str(extra_1) + ' not in both SAIF files! Error! Exiting...')
    sys.exit(1)

if len(extra_2):
    print(str(extra_2) + ' not in both SAIF files! Error! Exiting...')
    sys.exit(1)

cnt = 0
for key in dict_1.keys():
    if dict_1[key] != dict_2[key]:
        print('signal values for ' + str(key) + ' is not the same!')
        cnt += 1

print(f'Passed {cnt}/{len(dict_1)}')
