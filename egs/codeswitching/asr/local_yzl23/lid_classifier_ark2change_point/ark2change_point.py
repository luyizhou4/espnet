#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-12-30 12:45:10
# @Function:            
# @Last Modified time: 2020-01-07 20:41:56

import numpy as np
import kaldi_io 
import sys

#apply slide-window approximation, #window_size=3
def merge_frameids_approximation(frame_lids):
    new_frame_ids = []
    frame_num = len(frame_lids)
    assert frame_num >= 3
    for i in range(frame_num):
        left = frame_lids[i-1] if i > 0 else frame_lids[2]
        right = frame_lids[i+1] if i < (frame_num-1) else frame_lids[frame_num-3]
        mid = frame_lids[i]
        result = 0 if (mid+left+right) < 2 else 1
        new_frame_ids.append(result)

    frame_lids = new_frame_ids
    merged_lid_list = [frame_lids[0]]
    for lid in frame_lids:
        if lid == merged_lid_list[-1]:
            continue
        else:
            merged_lid_list.append(lid)
    return merged_lid_list  

# apply slide-window approximation, #window_size=3, TODO
def merge_frameids_approximation_ws(frame_lids, window_size=3):
    assert window_size > 0 and window_size % 2 == 1 # window_size with odd number is prefered
    frame_num = len(frame_lids)
    assert frame_num >= window_size
    cached_size = window_size // 2
    remained_size = window_size - cached_size
    new_frame_ids = frame_lids[remained_size:window_size] + frame_lids + frame_lids[-window_size:-remained_size]
    assert len(new_frame_ids) == frame_num + 2 * cached_size
    result_ids = []
    for i in range(frame_num):
        left = 0
        right = 0
        # currently, center point is (i+cached_size)
        for j in range(1, cached_size+1):
            left += new_frame_ids[i+cached_size-j]
        for j in range(1, cached_size+1):
            right += new_frame_ids[i+cached_size+j]    
        mid = new_frame_ids[i+cached_size]
        result = 0 if (mid+left+right) < 0.5 * window_size else 1
        result_ids.append(result)

    # here we merge language ids
    frame_lids = result_ids
    merged_lid_list = [frame_lids[0]]
    for lid in frame_lids:
        if lid == merged_lid_list[-1]:
            continue
        else:
            merged_lid_list.append(lid)
    return merged_lid_list  

def merge_frameids(frame_lids):
    merged_lid_list = [frame_lids[0]]
    for lid in frame_lids:
        if lid == merged_lid_list[-1]:
            continue
        else:
            merged_lid_list.append(lid)
    return merged_lid_list  

def main():
    # moe_coe_scp = "./exp/lid_classifier/transformer_layer6_lsm0.0_ep150/decode_greedy_decoding_mix20-dev_softmax/moe_coe.scp"
    # label_id = "./data/moe_json_data/tf_6l_lsm0.0/dev_mix20/label.id"
    # out_root = './exp/lid_classifier/tf6_lsm0.0_output_analysis/'

    moe_coe_scp = sys.argv[1]
    label_id = sys.argv[2]
    out_root = sys.argv[3]
    window_size = int (sys.argv[4])

    hyp_utt2lid = {}
    with open(moe_coe_scp, 'r') as fd, open(out_root+'hyp.utt2lid', 'w+') as w_fd:
        for line in fd.readlines():
            if not line.strip():
                continue
            uttid, path = line.split()
            frame_lids = np.argmax(kaldi_io.read_mat(path), axis=-1).tolist()
            # parse moe_coe into language change point
            assert len(frame_lids) > 0
            merged_lid_list = merge_frameids_approximation_ws(frame_lids, window_size)
            hyp_utt2lid[uttid] = merged_lid_list

            w_fd.write( uttid + ' ' + ' '.join( map(str, merged_lid_list)) + '\n' )
            w_fd.flush()

    ref_utt2lid = {}
    with open(label_id, 'r') as fd, open(out_root+'ref.utt2lid', 'w+') as w_fd:
        for line in fd.readlines():
            if not line.strip():
                continue
            tmp = line.split()
            uttid = tmp[0]
            char_lids = map(int, tmp[1:])
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