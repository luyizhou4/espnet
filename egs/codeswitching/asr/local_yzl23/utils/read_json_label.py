# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-10-08 15:36:36
# @Function:            
# @Last Modified time: 2019-12-08 21:11:19

import sys
import json

def main():
    in_json_file = sys.argv[1]
    out_txt_file = sys.argv[2]

    with open(in_json_file, 'r') as fd:
        with open(out_txt_file, 'w+') as w_fd:
            data = json.load(fd)
            uttid_list = list (data["utts"].keys())
            uttid_list.sort()
            print('There are totally %s utts'%(len(uttid_list)))
            for uttid in uttid_list:
                # note that <eos> is not removed
                rec_tokenid_list = data['utts'][uttid]["output"][0]["rec_tokenid"].split()
                rec_tokenid = ' '.join(rec_tokenid_list)
                w_fd.write(uttid + ' ' + rec_tokenid + '\n') 

if __name__ == '__main__':
    main()
