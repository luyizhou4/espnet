#!/bin/python
import argparse
import numpy as np
import sys


parser = argparse.ArgumentParser(description='Add random drop on phone alignment')
parser.add_argument('ali', type=str, help='ali file to add drop')
parser.add_argument('acc', type=str, help='acc file drop rate accroding to')
parser.add_argument('--out', type=str, default=None, help='output file')
args = parser.parse_args()

np.random.seed(1)
vob_dim=128

if args.out is None:
    outfile=sys.stdout
else:
    outfile=open(args.out, 'w')

uttid2acc={}
with open(args.acc, 'r') as fin:
    for line in fin:
        uttid, acc = line.split()[0], float(line.split()[1]) / 100
        uttid2acc[uttid] = acc

with open(args.ali, 'r') as fin:
    for line in fin:
        uttid, l_list = line.split()[0], np.array( [int(x) for x in line.split()[1:]])
        rate = 1 - uttid2acc[uttid]
        drop_idx =  np.random.permutation(len(l_list))[:int(len(l_list)*rate)]
        l_list[drop_idx] = [np.random.randint(low=0, high=vob_dim) for _ in range(len(drop_idx))]
        print(uttid, str(list(l_list)).replace(",","")[1:-1], flush=True, file=outfile)

if args.out is not None:
    outfile.close()

