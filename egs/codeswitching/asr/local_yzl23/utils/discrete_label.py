# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-09-24 17:26:47
# @Function:            
# @Last Modified time: 2019-09-24 18:52:04

import sys

def main():
    ref_file = sys.argv[1]
    out_file = sys.argv[2]

    with open(ref_file, 'r') as r_fd:
        with open(out_file, 'w+') as w_fd:
            for line in r_fd.readlines():
                if line.strip():
                    tmp = line.split()
                    uttid = tmp[0]
                    assert len(tmp) > 0
                    # ref text is empty
                    if len(tmp) == 0:
                        w_fd.write(uttid + '\n')
                        w_fd.flush()
                    else:
                        content = ' '.join(tmp[1:])
                        # parse content into word list
                        words_list = []
                        en_word = ''
                        for token in content:
                            # en token
                            if token <= 'z' and token != ' ' and token != '\n':
                                en_word += token
                            else:
                                if en_word: # space or endline
                                    words_list.append(en_word)
                                    en_word = ''
                                if token > 'z': # cn_word
                                    words_list.append(token)
                        if en_word:
                            words_list.append(en_word)
                        # write line
                        w_fd.write(uttid + ' ' + ' '.join(words_list) + '\n')
                        w_fd.flush()


if __name__ == '__main__': 
    main()
