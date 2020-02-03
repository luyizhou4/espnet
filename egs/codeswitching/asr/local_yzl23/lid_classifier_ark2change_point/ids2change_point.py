#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-12-30 12:45:10
# @Function:            
# @Last Modified time: 2019-12-31 13:56:05

import numpy as np
import sys

def main():
    label_id = sys.argv[1]
    out_file = sys.argv[2]
    MODE = int (sys.argv[3]) # > 0 to ignore last eos

    ref_utt2lid = {}
    with open(label_id, 'r') as fd, open(out_file, 'w+') as w_fd:
        for line in fd.readlines():
            if not line.strip():
                continue
            tmp = line.split()
            uttid = tmp[0]
            char_lids = list (map(int, tmp[1:])) 
            # ignore final <eos>
            if MODE > 0:
                char_lids = char_lids[:-1]
            char_lids = [0 if lid < 1000 else 1 for lid in char_lids]
            # parse char_lids into language change point
            assert len(char_lids) > 0
            merged_lid_list = [char_lids[0]]
            for lid in char_lids:
                if lid == merged_lid_list[-1]:
                    continue
                else:
                    merged_lid_list.append(lid)
            ref_utt2lid[uttid] = merged_lid_list

            w_fd.write( uttid + ' ' + ' '.join( map(str, merged_lid_list)) + '\n' )
            w_fd.flush()


if __name__ == '__main__':
    main()