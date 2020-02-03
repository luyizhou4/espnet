#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-12-16 15:53:39
# @Function: convert ctm file to language chunk, this script simply merges adjacent words into language chunk            
# @Last Modified time: 2020-01-03 14:53:11

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

class utt_obj(object):

    def __init__(self, uttid, word_start_time, word_duration, word_content):
        '''
            deal with kaldi ctm data
        '''
        self.utt_obj_init(uttid)
        # initial data process
        self.add_new_record(word_start_time, word_duration, word_content)

    def utt_obj_init(self, uttid):
        self.uttid = uttid
        self.chunk_list = [] # list of dict { LID: [start_time, end_time] } 
        self.current_chunk = None

    def add_new_record(self, word_start_time, word_duration, word_content):
        assert word_content # word_content can not be empty
        if re.match("\[[SNTP]\]", word_content): # filter out [S|N|T|P]
            return
        current_lang = "EN" if word_content[0] < 'z' else 'CN'
        if self.current_chunk:
            if current_lang == self.current_chunk["lid"]: # language chunk continues
                # merges chunk, NOTE that we ignore 'silence chunk' here
                self.current_chunk["end"] = word_start_time + word_duration
            else: # language chunk ends
                self.chunk_list.append(self.current_chunk)
                self.current_chunk = {"lid": current_lang,
                                "start": word_start_time,
                                "end": word_start_time+word_duration}
        else:
            # add a new chunk
            self.current_chunk = {"lid": current_lang,
                                "start": word_start_time,
                                "end": word_start_time+word_duration}

    def get_lang_chunk_contents(self):
        # NOTE: append the last chunk
        self.chunk_list.append(self.current_chunk)
        self.current_chunk = None
        out_contents = []
        for chunk in self.chunk_list:
            content = "{} {} {:.2f} {:.2f} {:.2f}\n".format(self.uttid, chunk["lid"],\
                                            chunk["start"], chunk["end"], chunk["end"]-chunk["start"])
            out_contents.append(content)
        return out_contents

    def get_label(self, num_frames):
        # NOTE: append the last chunk
        self.chunk_list.append(self.current_chunk)
        self.current_chunk = None
        label_ids = []
        start, end = 0, 0
        num_chunks = len(self.chunk_list)
        for i in range(num_chunks):
            end = round( (self.chunk_list[i]["end"] + self.chunk_list[i+1]["start"]) / 2 / 0.01) \
                    if i != (num_chunks - 1) else num_frames
            # now we have[start, end] chunk here
            chunk_label = 0 if self.chunk_list[i]['lid'] == 'EN' else 1
            label_ids.extend( (end - start) * [chunk_label])
            start = end

        assert len(label_ids) == num_frames, "{} len(label_ids): {} != num_frames: {}".format(self.uttid, len(label_ids), num_frames)
        label_ids = self.subsampling_label(label_ids)
        label_ids = map(str, label_ids)
        return self.uttid + " " + " ".join(label_ids)

    def subsampling_label(self, label_ids):
        target_len = ( (len(label_ids) - 1) // 2 - 1 ) // 2
        label_ids = label_ids[:-2:2][:-2:2]
        assert len(label_ids) == target_len
        return label_ids

    # write out scp & ark files for ground truth labels
    def get_out_mat(self, num_frames):
        # NOTE: append the last chunk
        self.chunk_list.append(self.current_chunk)
        self.current_chunk = None
        label_ids = []
        start, end = 0, 0
        num_chunks = len(self.chunk_list)
        for i in range(num_chunks):
            end = round( (self.chunk_list[i]["end"] + self.chunk_list[i+1]["start"]) / 2 / 0.01) \
                    if i != (num_chunks - 1) else num_frames
            # now we have[start, end] chunk here
            chunk_label = 0 if self.chunk_list[i]['lid'] == 'EN' else 1
            label_ids.extend( (end - start) * [chunk_label])
            start = end

        assert len(label_ids) == num_frames, "{} len(label_ids): {} != num_frames: {}".format(self.uttid, len(label_ids), num_frames)
        label_ids = self.subsampling_label(label_ids)
        # label ids to mat
        return self.label_ids2mat(label_ids)

    def label_ids2mat(self, label_ids):
        T = len(label_ids)
        out_mat = np.zeros((T, 2))
        for i in range(T):
            out_mat[i, label_ids[i]] = 1.0
        return out_mat


def get_utt2num_frames(utt2num_frames_path):
    utt2num_frames = {}
    with open(utt2num_frames_path, 'r') as fd:
        for line in fd.readlines():
            if not line.strip():
                continue
            tmp = line.split()
            utt2num_frames[tmp[0]] = int (tmp[1])

    return utt2num_frames

def output_lang_chunk():
    log_interval = 10e5
    root_path = "./data"
    ctm_path = os.path.join(root_path, "mix200.ctm")
    out_path = os.path.join(root_path, "lang.chunk")

    with open(ctm_path, 'r', encoding='utf-8') as fd, open(out_path, 'w+') as w_fd:
        last_utterance = None
        current_utt_obj = None
        for line_count, line in enumerate(fd.readlines()):
            if not line.strip():
                continue
            if (line_count+1) % log_interval == 0:
                print("Currently deal with line: %d"%(line_count+1))
            tmp = line.split()
            assert len(tmp) == 6 # 6 fields
            uttid, _, word_start_time, word_duration, word_content, _ = tmp
            if uttid == last_utterance: # still in the same utterance
                # add time stamp
                current_utt_obj.add_new_record(float (word_start_time), 
                                    float (word_duration), word_content)
            else:
                last_utterance = uttid
                if current_utt_obj:
                    # write out current utterance language chunk
                    out_contents = current_utt_obj.get_lang_chunk_contents()
                    for out_content in out_contents:
                        w_fd.write(out_content)
                    w_fd.flush()
                # deal with a new utterance
                current_utt_obj = utt_obj(uttid, float (word_start_time), 
                                    float (word_duration), word_content)

        # write out current utterance language chunk
        out_contents = current_utt_obj.get_lang_chunk_contents()
        for out_content in out_contents:
            w_fd.write(out_content)
        w_fd.flush()    

def visualize_chunks():
    chunk_file = './data/lang.chunk.store'
    data = {"CN": [], "EN": []}
    with open(chunk_file, 'r') as fd:
        for line in fd.readlines():
            if not line.strip():
                continue
            _, lid, _, _, duration = line.split()
            data[lid].append(float (duration))

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set(color_codes=True)
    plt.subplot(2, 1, 1)
    sns.distplot(data["CN"])
    plt.subplot(2, 1, 2)
    sns.distplot(data["EN"])
    plt.savefig('chunk_duration.png')

    total_cn_time = np.sum(np.array(data["CN"]))
    total_en_time = np.sum(np.array(data["EN"]))
    total_time = total_cn_time + total_en_time
    print("Total CN chunk time: {:.2f}h, {:.2f}%, EN chunk {:.2f}h, {:.2f}%".format(total_cn_time/3600, total_cn_time / total_time, 
                total_en_time/3600, total_en_time / total_time))
    
    # Result:
    # Total CN chunk time: 103.00h, 0.69%, EN chunk 46.07h, 0.31%

def output_labels():
    log_interval = 10e5
    root_path = sys.argv[1]
    utt2num_frames_path = os.path.join(root_path, "utt2num_frames")
    ctm_path = os.path.join(root_path, "ctm")
    root_path = os.path.abspath(root_path)
    out_path = "ark:| copy-feats ark:- ark,scp:{}/moe_coe.ark,{}/moe_coe.scp".format(root_path, root_path)
    # out_path = "{}/moe_coe.ark".format(root_path)

    utt2num_frames = get_utt2num_frames(utt2num_frames_path)

    with open(ctm_path, 'r', encoding='utf-8') as fd, kaldi_io.open_or_fd(out_path, 'wb') as w_fd:
        last_utterance = None
        current_utt_obj = None
        for line_count, line in enumerate(fd.readlines()):
            if not line.strip():
                continue
            if (line_count+1) % log_interval == 0:
                print("Currently deal with line: %d"%(line_count+1))
            tmp = line.split()
            assert len(tmp) == 6 # 6 fields
            uttid, _, word_start_time, word_duration, word_content, _ = tmp
            if uttid == "mix200h_T0455G0188_S02020331": # this utterance has unknown problems
                continue
            if uttid == last_utterance: # still in the same utterance
                # add time stamp
                current_utt_obj.add_new_record(float (word_start_time), 
                                    float (word_duration), word_content)
            else:
                last_utterance = uttid
                if current_utt_obj:
                    # write out current utterance language chunk
                    mat = current_utt_obj.get_out_mat(utt2num_frames[current_utt_obj.uttid])
                    kaldi_io.write_mat(w_fd, mat, current_utt_obj.uttid)
                    w_fd.flush()
                # deal with a new utterance
                current_utt_obj = utt_obj(uttid, float (word_start_time), 
                                    float (word_duration), word_content)

        # write out current utterance language chunk
        mat = current_utt_obj.get_out_mat(utt2num_frames[current_utt_obj.uttid])
        kaldi_io.write_mat(w_fd, mat, current_utt_obj.uttid)
        w_fd.flush()


if __name__ == '__main__':
    output_labels()

