#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2020-02-11 14:46:36
# @Function:            
# @Last Modified time: 2020-02-11 15:21:46

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import kaldi_io

def main():
    filename = sys.argv[1]
    out_path = sys.argv[2]
    lang_emb_start = int (sys.argv[3])
    lang_emb_end = int (sys.argv[4])
    emb = kaldi_io.read_mat(filename)
    emb = np.transpose(emb, (1,0))[lang_emb_start:lang_emb_end, :]
    print(emb.shape)
    plt.matshow(emb, fignum=None)
    plt.savefig(out_path + '.png')

if __name__ == '__main__':
    main()