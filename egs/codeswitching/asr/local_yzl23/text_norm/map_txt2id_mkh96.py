import sys

if len(sys.argv) != 5:
    print('Usage:', sys.argv[0], '<text file> <cn vocab> <en lexicon> <en_vocab_size>')
    exit(0)

mix_txt = sys.argv[1]
cn_vocab = sys.argv[2]
en_lex = sys.argv[3]
bpe = int(sys.argv[4])

dt_cn = {}
unk = str(bpe)
# cn dict
with open(cn_vocab, 'r', encoding='utf-8') as f:
    i = bpe + 1
    for line in f:
        dt_cn[line[0]] = str(i)
        i += 1

dt_en = {}
# en lexicon
with open(en_lex, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.split()
        dt_en[line[0]] = line[1:]

# map text to id
with open(mix_txt, 'r', encoding='utf-8') as f:
    for line in f:
        ids = []
        word = ''
        for c in line:
            if c <= 'z' and c != ' ' and c != '\n':
                word += c
            else:
                # NOTE last '\n' 
                if len(word) > 0:
                    if dt_en.get(word) is None:
                        ids += [unk]
                    else:
                        ids += dt_en[word]
                    word = ''
                if c > 'z':
                    if dt_cn.get(c) is None:
                        ids += [unk]
                    else:
                        ids += [dt_cn[c]]
        if len(word) > 0:
            if dt_en.get(word) is None:
                ids += [unk]
            else:
                ids += dt_en[word]
        print(' '.join(ids))

