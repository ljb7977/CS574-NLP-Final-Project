import re


tagged_f = open('tagged_train.en', 'r', encoding='utf-8')
plain_filename = 'train.en'
plain_file = open(plain_filename, 'w', encoding='utf-8')
bulk = tagged_f.read()
blocks = re.compile(r"\n{2,}").split(bulk)
blocks = list(filter(None, blocks))
sentences = []
cnt = 0
for block in blocks:
    tokens = []
    for line in block.splitlines():
        attr_list = line.split('\t')
        tokens.append(attr_list[1])
    cnt += 1
    print(cnt)
    for i, token in enumerate(tokens):
        if i != 0:
            plain_file.write(' ')
        plain_file.write(token)
    plain_file.write('\n')
plain_file.close()
tagged_f.close()


