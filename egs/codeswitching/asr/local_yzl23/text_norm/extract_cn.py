# get mandarin word frequency
import sys

if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], '<file>')
    exit(0)

fname = sys.argv[1]
cn2freq = {}
with open(fname, 'r', encoding='utf-8') as f:
    for line in f:
        for c in line:
            if c > 'z':
                if cn2freq.get(c) is None:
                    cn2freq[c] = 1
                else:
                    cn2freq[c] += 1
    lt = sorted(cn2freq.items(), key=lambda x: x[1], reverse=True)
    for k, v in lt:
        print(k, v)
