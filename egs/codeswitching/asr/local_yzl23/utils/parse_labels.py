# -*- coding: utf-8 -*-
# @Author: luyizhou4
# @Date:   2019-09-23 20:47:40
# @Function:            
# @Last Modified time: 2019-12-09 14:13:45

import sys

def main():
    labels_file = sys.argv[1]
    result_file = sys.argv[2]
    vocab = sys.argv[3]
    ignored_tokens = sys.argv[4] # "0 1000 4005 4006 4007"
    IGNORED_TOKENS_LIST = list(map(int, ignored_tokens.split('_')))
    EN_NUM = 1000 # used for dividing CN & EN words

    id2char = {}
    char2id = {}
    # get char2id & id2char
    with open(vocab, 'r', encoding='utf-8') as fd:
        for i, line in enumerate(fd.readlines()):
            if line.strip():
                char = line.strip()
                id2char[i+1] = char
                char2id[char] = i+1
    # read result.txt and write out result.word.txt
    with open(labels_file, 'r', encoding='utf-8') as r_fd:
        with open(result_file, 'w+') as w_fd:
            for line in r_fd.readlines():
                if line.strip():
                    # empty content
                    tmp = line.split()
                    assert len(tmp) > 0
                    if len(tmp) == 1:
                        # write line
                        w_fd.write(tmp[0] + '\n')
                        w_fd.flush()
                    else:
                        uttid = tmp[0]
                        content = list(map(int, tmp[1:]))
                        # parse content
                        words_list = []
                        en_word = ''
                        for label_id in content:
                            if label_id in IGNORED_TOKENS_LIST:
                                continue
                            if label_id > EN_NUM: # cn_word
                                if en_word:
                                    words_list.append(en_word)
                                    en_word = ''
                                words_list.append(id2char[label_id])
                            elif label_id < EN_NUM: # en
                                token = id2char[label_id]
                                if token.startswith('▁'): # en_word header
                                    if en_word:
                                        words_list.append(en_word)
                                        en_word = ''
                                    # whatif we only get a '▁'
                                    en_word = token
                                else:
                                    en_word = en_word + token

                        # en_word in sentence end
                        if en_word:
                            words_list.append(en_word)
                            en_word = ''

                        # remove '▁'
                        for i, token in enumerate(words_list):
                            if '▁' in token:
                                words_list[i] = token.replace('▁', '')

                        # write line
                        w_fd.write(uttid + ' ' + ' '.join(words_list) + '\n')
                        w_fd.flush()

if __name__ == '__main__':
    main()