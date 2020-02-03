# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-10-05 20:23:17
# @Function:            
# @Last Modified time: 2019-10-09 17:06:00

import sys
import os
import json

class utterance_obj(object):
    def __init__(self, uttid, feat_path=None, feat_shape=None, feat_name='input1', 
            label_name="target1", label_shape=None, text=None, tokenid=None,
            utt2spk=None):
        '''object that contains required json properties
        Args:
            follows ESPnet json setup            
        '''
        self.uttid = uttid

        # input_list
        self.feat_path = feat_path
        self.feat_name = feat_name
        self.feat_shape = feat_shape
        
        # output list 
        self.label_name = label_name
        self.label_shape = label_shape
        self.text = text
        self.tokenid = tokenid

        # utt2spk
        self.utt2spk = utt2spk

        # final list
        self.input = None
        self.output = None

    def set_input(self):
        if self.feat_name and self.feat_path and self.feat_shape:
            self.input = [{
                        "feat": self.feat_path, 
                        "name": self.feat_name, 
                        "shape": self.feat_shape,
                        }]
    
    def set_output(self):
        # check if self.output exists
        if self.label_name and self.label_shape and self.text and self.tokenid:
            self.output = [{
                        "name": self.label_name, 
                        "shape": self.label_shape, 
                        "text": self.text, 
                        "tokenid": self.tokenid,
                        }]

    def check_all_filled(self):
        if self.input and self.output and self.utt2spk:
            return True
        else:
            return False
    
    # return the obj as a map
    def get_all(self):
        self.set_input()
        self.set_output()
        if self.check_all_filled():
            return True, {"input":self.input, "output":self.output, "utt2spk": self.utt2spk}
        else:
            return False, None

def main():
    # root_dir shall contains the following
    # feats.scp, featlen, text, label.id, utt2spk
    root_dir = sys.argv[1] 
    feats_dim = sys.argv[2] # eg. 80d fbank --> 80
    label_dim = sys.argv[3] # 4006, 0 is reserved as <blk>, and the final token is <sos>/<eos>
    constrain_params = sys.argv[4] # featlen constrain, token_num constrain. 1200_50, to disable: 0_0
    utt_obj_dict = {}

    max_featlen  = int (constrain_params.split('_')[0])
    max_token_num = int (constrain_params.split('_')[1])

    # input
    with open(os.path.join(root_dir, 'feats.scp'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                id_, feat_path = line.split()
                utt_obj_dict[id_] = utterance_obj(id_, feat_path=feat_path)

    with open(os.path.join(root_dir, 'utt2num_frames'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                id_, featlen = line.split()
                featlen = int (featlen)
                if id_ in utt_obj_dict:
                    if max_featlen > 0 and featlen > max_featlen: # featlen_constrained & exceed
                        continue
                    else:
                        utt_obj_dict[id_].feat_shape = [featlen, int (feats_dim)]

    # output part
    with open(os.path.join(root_dir, 'text'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                tmp = line.split()
                id_ = tmp[0]
                if id_ in utt_obj_dict:
                    utt_obj_dict[id_].text = " ".join(tmp[1:])

    with open(os.path.join(root_dir, 'label.id'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                tmp = line.split()
                id_ = tmp[0]
                if id_ in utt_obj_dict:
                    if max_token_num > 0 and (len(tmp) - 1 > max_token_num): # token_constrained & exceed
                        continue
                    else:
                        utt_obj_dict[id_].tokenid = " ".join(tmp[1:])
                        utt_obj_dict[id_].label_shape = [len(tmp) - 1, int (label_dim)]

    with open(os.path.join(root_dir, 'utt2spk'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                id_, spk = line.split()
                if id_ in utt_obj_dict:
                    utt_obj_dict[id_].utt2spk = spk

    utt_dict = {}
    for uttid, uttobj in utt_obj_dict.items():
        # put all 'good' uttobj into utt_dict
        checked, res = uttobj.get_all()
        if checked:
            utt_dict[uttid] = res

    print("Remain utts: %d"%(len(utt_dict)))
    all_dict = {'utts': utt_dict}
    with open(os.path.join(root_dir, 'data.json'), 'w+', encoding='utf-8') as fd:
        json.dump(all_dict, fd, indent=4, ensure_ascii=False, \
                    sort_keys=True, separators=(",", ": "))


if __name__ == '__main__':
    main()
