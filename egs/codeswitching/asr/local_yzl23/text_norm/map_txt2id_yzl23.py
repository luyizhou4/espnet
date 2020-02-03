'''
     convert text to id: 
        english words will be parsed by sentencepiece, 
        note that 0 is reserved for <unk> in sentencepiece, however, we need 0 to be <blank> in ESPnet JCA mode
        thus, we cover all characters in spm_train with --character_coverage=1.0
        we set <unk>=#en_vocab_size for chinese character use, 
        and chinese character start from <en_vocab_size> + 1
'''

import sys
import sentencepiece as spm

if len(sys.argv) != 5:
    print('Usage:', sys.argv[0], '<text file> <cn_vocab> <en_vocab_size> <bpe_model>')
    exit(0)

TEXT_FILE = sys.argv[1]
cn_vocab = sys.argv[2]
en_vocab_size = int(sys.argv[3])
bpe_model = sys.argv[4]

sp = spm.SentencePieceProcessor()
sp.Load(bpe_model)
# sp.EncodeAsIds("This is a test")

dt_cn = {}
unk = str(en_vocab_size)
# cn dict
with open(cn_vocab, 'r', encoding='utf-8') as f:
    i = en_vocab_size + 1
    for line in f:
        dt_cn[line[0]] = str(i)
        i += 1

# map text to id
with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.split()
        assert len(tmp) > 0
        contents = ' '.join(tmp[1:])
        ids = [tmp[0]]
        word = ''
        for c in contents:
            if c <= 'z' and c != ' ' and c != '\n': # en word
                word += c
            else:
                # parse en word
                if word:
                    ids += map(str, sp.EncodeAsIds(word))
                    word = ''
                if c > 'z': # cn word
                    if dt_cn.get(c) is None:
                        ids += [unk]
                    else:
                        ids += [dt_cn[c]]
        if word:
            ids += map(str, sp.EncodeAsIds(word))
            word = ''
        print(' '.join(ids))

