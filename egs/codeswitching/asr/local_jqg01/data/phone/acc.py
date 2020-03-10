import sys
import numpy as np


reff=sys.argv[1]
hybf=sys.argv[2]

def read_f(fname):
    res={}
    with open(fname, 'r') as fin:
        for line in fin:
            uttid, ali = line.split()[0], line.split()[1:]
            res[uttid]=np.array([ int(x) for x in ali])
    return res

ref=read_f(reff)
hyb=read_f(hybf)

def cal_acc(x, y):
    assert abs(len(x) - len(y)) < 3
    acc_arr = x == y
    return np.count_nonzero(acc_arr) / len(x)


assert len(ref) == len(hyb)
acc_cnt=0.0
cnt=0
for k in ref:
    acc = cal_acc(ref[k], hyb[k])
    cnt+=1
    acc_cnt+=acc
    print(k, acc)

print('Total: cnt{} acc{}'.format(str(cnt), str(acc_cnt/cnt)))

