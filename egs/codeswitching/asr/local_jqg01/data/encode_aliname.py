import sys
import json

tr_name = sys.argv[1]
dev_name = sys.argv[2]
out_file = sys.argv[3]

with open(tr_name, 'r') as f:
    tr_data = json.load(f)['utts']
with open(dev_name, 'r') as f:
    dev_data = json.load(f)['utts']
data = list(tr_data.keys())+list(dev_data.keys())

idx = 0
with open(out_file, 'w') as f:
    for d in data:
        f.write(d+" "+str(idx)+'\n')
        idx += 1


