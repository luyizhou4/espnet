#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-12-16 15:53:39
# @Function: convert ctm file to language chunk, this script simply merges adjacent words into language chunk            
# @Last Modified time: 2020-01-09 16:02:42

''' output file format example as below:

    uttid start_time end_time language
    mix200h_T0001G0001_S01010002 0.0 0.52 CN
    mix200h_T0001G0001_S01010002 0.52 1.44 EN

    Note:
    1. filter out [S|N|T|P], can consider as sil and ignore
    2. there might be 'sil' anywhere

'''

import re
import os
import sys
import kaldi_io
import numpy as np

def get_utt2num_frames(utt2num_frames_path):
    utt2num_frames = {}
    with open(utt2num_frames_path, 'r') as fd:
        for line in fd.readlines():
            if not line.strip():
                continue
            tmp = line.split()
            utt2num_frames[tmp[0]] = int (tmp[1])

    return utt2num_frames

# utt2num_frames subsampling
def utt2num_frames_subsampling(utt2num_frames):
    for k in utt2num_frames.keys():
        utt2num_frames[k] = ((utt2num_frames[k] - 1) // 2 - 1) // 2
    return utt2num_frames

def get_out_mat(num_frames, language_type):
    out_mat = np.zeros((num_frames, 2))
    out_mat[:, language_type] = 1.0
    return out_mat


# target_len = ( (len(label_ids) - 1) // 2 - 1 ) // 2
def output_labels():
    log_interval = 10e5
    root_path = sys.argv[1]
    language_type = int (sys.argv[2]) # 0 for english, 1 for chinese
    utt2num_frames_path = os.path.join(root_path, "utt2num_frames")
    root_path = os.path.abspath(root_path)
    out_path = "ark:| copy-feats ark:- ark,scp:{}/moe_coe.ark,{}/moe_coe.scp".format(root_path, root_path)
    # out_path = "{}/moe_coe.ark".format(root_path)
    moe_coe_len = os.path.join(root_path, "moe_coe.len")

    utt2num_frames = get_utt2num_frames(utt2num_frames_path)
    utt2num_frames = utt2num_frames_subsampling(utt2num_frames)
    with open(moe_coe_len, 'w+') as len_fd, kaldi_io.open_or_fd(out_path, 'wb') as w_fd:
        for uttid, num_frames in utt2num_frames.items():
            len_fd.write("{} {}\n".format(uttid, num_frames))
            len_fd.flush()

            # write out fake language ids
            mat = get_out_mat(num_frames, language_type)
            kaldi_io.write_mat(w_fd, mat, uttid)
            w_fd.flush()


if __name__ == '__main__':
    output_labels()

