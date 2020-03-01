import kaldiio
import sys
import numpy as np
from kaldiio import WriteHelper


alif = sys.argv[1]
phonetab = sys.argv[2]
outf = sys.argv[3]

ptab = {}
dim = 0
with open(phonetab, 'r') as phnin:
    for line in phnin:
        phn, pid = line.split()[0], line.split()[1]
        ptab[phn] = int(pid)
        dim += 1

uid2onehot = {}
with open(alif, 'r') as fin:
    for line in fin:
        uid, ali = line.split()[0], line.split()[1:]
        l_list = []
        for l in ali:
            l_list.append(ptab[l])
        l_list = np.array(l_list, dtype='int')
        one_hot = np.zeros((l_list.size, dim))
        one_hot[np.arange(l_list.size), l_list] = 1
        uid2onehot[uid] = one_hot

outfname = 'ark,scp:{}.ark,{}.scp'.format(outf, outf)
with WriteHelper(outfname,compression_method=2) as writer:
    for k,v in uid2onehot.items():
        writer(uid, one_hot)
        




