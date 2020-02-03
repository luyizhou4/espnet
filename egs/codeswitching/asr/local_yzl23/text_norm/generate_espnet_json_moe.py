# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-12-08 18:23:17
# @Function: generate LID-label style json file           
# @Last Modified time: 2019-12-26 12:41:03

import sys
import os
import json

class utterance_obj(object):
    def __init__(self, uttid, moe_mode, feat_path=None, feat_shape=None, feat_name='input1', 
            label_name="target1", label_shape=None, text=None, tokenid=None,
            utt2spk=None, moe_path=None, moe_shape=None, moe_name='moe_coe'):
        '''object that contains required json properties
        Args:
            follows ESPnet json setup            
        '''
        self.uttid = uttid
        self.moe_mode = moe_mode

        # input_list
        self.feat_path = feat_path
        self.feat_shape = feat_shape
        self.feat_name = feat_name

        # moe list
        self.moe_path = moe_path
        self.moe_shape = moe_shape
        self.moe_name = moe_name
        
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
        moe_content_full = (self.moe_name and self.moe_path and self.moe_shape)
        feat_content_full = (self.feat_name and self.feat_path and self.feat_shape)
        if self.moe_mode:
            if moe_content_full and feat_content_full:
                self.input = [{
                            "feat": self.feat_path, 
                            "name": self.feat_name, 
                            "shape": self.feat_shape,
                            },{
                            "feat": self.moe_path,
                            "name": self.moe_name,
                            "shape": self.moe_shape
                            }]
        else:
            if feat_content_full:
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


def get_lid_tokens(mode, vocab_dim):
    if mode == 0:
        return None, vocab_dim + 2 # blank, <sos>/<eos>
    elif mode == 1 or mode == 2:
        # lid before/after, 0 for blank
        # vocab_dim+1 for <EN>, +2 for <CN>, +3 for <sos>/<eos>
        return [vocab_dim+1, vocab_dim+2], vocab_dim+4
    elif mode == 3:
        # change points, vocab_dim+1 for <CP>, +2 for <sos>/<eos>
        return vocab_dim+1, vocab_dim+3
    else:
        raise Exception('Not supported mode: %s'%(str(mode)))

def get_label_ids(label_ids, mode=0, divider=None, lid_tokens=None):
    ''' Get required label ids, 
        mode 0: directly return
        mode 1: lid before label ids
        mode 2: lid after label ids
        mode 3: add language change points
    '''
    assert mode in [0, 1, 2, 3]
    if mode == 0:
        return label_ids
    elif mode == 1:
        # lid before
        return get_lidbf_label_ids(label_ids, divider, lid_tokens)
    elif mode == 2:
        # lid after
        return get_lidaf_label_ids(label_ids, divider, lid_tokens)
    elif mode == 3:
        # lid change point
        return get_lidcp_label_ids(label_ids, divider, lid_tokens)
    else:
        raise Exception('Not supported mode: %s'%(str(mode)))

def get_lidbf_label_ids(label_ids, divider, lid_tokens):
    ''' Given label ids, return lidbf label_ids
        e.g. "english words 中 文" ==> "<EN> english words <CN> 中 文"
        :param label_ids: list of int, label tokens
        :param divider: unk token id, used as a divider
        :lid_tokens: list of int tokens ids for <EN> and <CN>
    '''
    # for empty utterance, directly return
    assert len(lid_tokens) == 2 # currently we only support CN/EN codeswitching mode
    assert divider > 0
    if len(label_ids) == 0:
        return label_ids
    prev_lang_state = 0 if label_ids[0] < divider else 1 # 0 for en, 1 for cn
    out_label_ids = [lid_tokens[prev_lang_state]] # add head language tokens
    
    for label_token in label_ids:
        lang_state = 0 if label_token < divider else 1
        if lang_state == prev_lang_state:
            out_label_ids.append(label_token)
        else:
            out_label_ids.extend([lid_tokens[lang_state], label_token])
            # update current language state
            prev_lang_state = lang_state
    return out_label_ids

def get_lidaf_label_ids(label_ids, divider, lid_tokens):
    ''' Given label ids, return lid_after label_ids
        e.g. "english words 中 文" ==> "english words <EN> 中 文 <CN>"
        :param label_ids: list of int, label tokens
        :param divider: unk token id, used as a divider
        :lid_tokens: list of int tokens ids for <EN> and <CN>
    '''
    # currently we only support CN/EN codeswitching mode
    assert len(lid_tokens) == 2 
    assert divider > 0
    # for empty utterance, directly return
    if len(label_ids) == 0:
        return label_ids
    
    out_label_ids = []
    for index, label_token in enumerate(label_ids):
        lang_state = 0 if label_token < divider else 1
        # check if it's time to emit language tokens
        if index == len(label_ids)-1:
            emit_status = True
        elif bool(label_token // divider) != bool(label_ids[index+1] // divider):
            emit_status = True
        else:
            emit_status = False

        if emit_status:
            out_label_ids.extend([label_token, lid_tokens[lang_state]])
        else:
            out_label_ids.append(label_token)

    return out_label_ids

def get_lidcp_label_ids(label_ids, divider, lid_tokens):
    ''' Given label ids, return lid_change_point label_ids
        e.g. "english words 中 文" ==> "english words <CP> 中 文"
        :param label_ids: list of int, label tokens
        :param divider: unk token id, used as a divider
        :lid_tokens: <CP> token id
    '''
    assert divider > 0
    assert lid_tokens > 0
    # for empty utterance, directly return
    if len(label_ids) == 0:
        return label_ids
    
    out_label_ids = []
    for index, label_token in enumerate(label_ids):
        lang_state = 0 if label_token < divider else 1
        # check if it's time to emit language tokens
        if index == len(label_ids)-1: # avoid index+1 out of range
            emit_status = False
        elif bool(label_token // divider) != bool(label_ids[index+1] // divider):
            emit_status = True
        else:
            emit_status = False

        if emit_status:
            out_label_ids.extend([label_token, lid_tokens])
        else:
            out_label_ids.append(label_token)

    return out_label_ids


def main():
    # root_dir shall contains the following
    # feats.scp, featlen, text, label.id, utt2spk
    root_dir = sys.argv[1] 
    FEATS_DIM = int(sys.argv[2]) # eg. 80d fbank --> 80
    VOCAB_DIM = int(sys.argv[3]) # 4004
    CONSTRAIN_PARAMS = sys.argv[4] # featlen constrain, token_num constrain. 1200_50, to disable: 0_0
    MAX_FEATLEN  = int(CONSTRAIN_PARAMS.split('_')[0])
    MAX_TOKEN_NUM = int(CONSTRAIN_PARAMS.split('_')[1])
    UNK = int(sys.argv[5]) # <unk> id, 1000 by default. This is used as a divider of language
    MODE = int(sys.argv[6]) # lid mode, 0 for normal, 1 for lid-bf, 2 for lid-af, 3 for lid-cp
    MOE_MODE = True if int(sys.argv[7]) > 0 else False
    print("MOE mode: " + str(MOE_MODE))

    json_tags = ['', '_lidbf', '_lidaf', '_lidcp']
    utt_obj_dict = {}
    # input
    with open(os.path.join(root_dir, 'feats.scp'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                id_, feat_path = line.split()
                utt_obj_dict[id_] = utterance_obj(id_, MOE_MODE, feat_path=feat_path)
    print("Total utts in feats.scp: %d"%(len(utt_obj_dict)))

    with open(os.path.join(root_dir, 'utt2num_frames'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                id_, featlen = line.split()
                featlen = int (featlen)
                if id_ in utt_obj_dict:
                    if MAX_FEATLEN > 0 and featlen > MAX_FEATLEN: # featlen_constrained & exceed
                        continue
                    else:
                        utt_obj_dict[id_].feat_shape = [featlen, FEATS_DIM]

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
                label_ids = list(map(int, tmp[1:]))
                lid_tokens, OUTPUT_LABEL_DIM = get_lid_tokens(MODE, VOCAB_DIM)
                label_ids = get_label_ids(label_ids, mode=MODE, divider=UNK, lid_tokens=lid_tokens)
                if id_ in utt_obj_dict:
                    if MAX_TOKEN_NUM > 0 and (len(label_ids) > MAX_TOKEN_NUM): # token_constrained & exceed
                        continue
                    else:
                        utt_obj_dict[id_].tokenid = " ".join( map(str, label_ids) )
                        utt_obj_dict[id_].label_shape = [len(label_ids), OUTPUT_LABEL_DIM]

    with open(os.path.join(root_dir, 'utt2spk'), 'r', encoding="utf-8") as fd:
        for line in fd.readlines():
            if line.strip():
                id_, spk = line.split()
                if id_ in utt_obj_dict:
                    utt_obj_dict[id_].utt2spk = spk

    if MOE_MODE:
        with open(os.path.join(root_dir, 'moe_coe.scp'), 'r', encoding="utf-8") as fd:
            for line in fd.readlines():
                if line.strip():
                    id_, path = line.split()
                    if id_ in utt_obj_dict:
                        utt_obj_dict[id_].moe_path = path
        with open(os.path.join(root_dir, 'moe_coe.len'), 'r', encoding="utf-8") as fd:
            for line in fd.readlines():
                if line.strip():
                    id_, frames = line.split()
                    if id_ in utt_obj_dict:
                        utt_obj_dict[id_].moe_shape = [int (frames), 2]

    utt_dict = {}
    for uttid, uttobj in utt_obj_dict.items():
        # put all 'good' uttobj into utt_dict
        checked, res = uttobj.get_all()
        if checked:
            utt_dict[uttid] = res

    
    tag = json_tags[MODE]
    json_name = os.path.join(root_dir, 'data%s.json'%(tag))
    all_dict = {'utts': utt_dict}
    with open(json_name, 'w+', encoding='utf-8') as fd:
        json.dump(all_dict, fd, indent=4, ensure_ascii=False, \
                    sort_keys=True, separators=(",", ": "))
    print("Remain utts: %d"%(len(utt_dict)))
    with open(json_name + '.count', 'w+', encoding='utf-8') as fd:
        fd.write("Remain utts: %d\n"%(len(utt_dict)))


if __name__ == '__main__':
    main()
