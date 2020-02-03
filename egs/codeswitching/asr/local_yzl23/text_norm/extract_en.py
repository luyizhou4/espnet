import sys

if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], '<file>')
    exit(0)

fname = sys.argv[1]
dt = {}
with open(fname, 'r', encoding='utf-8') as f:
    for line in f:
        word = ''
        found = False
        for c in line:
            if c <= 'z' and c != ' ' and c != '\n':
                word += c
            elif len(word) > 0:
                found = True
                print(word, end=' ')
                word = ''
        if len(word) > 0:
            found = True
            print(word, end=' ')
        if found: print()

