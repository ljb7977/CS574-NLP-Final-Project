dev_tok_f = open('../dev.kr', 'r', encoding='utf-8')
test_tok_f = open('../test.kr', 'r', encoding='utf-8')
train_tok_f = open('../train.kr', 'r', encoding='utf-8')
dev_f = open('dev_origin.kr', 'w', encoding='utf-8')
test_f = open('test_origin.kr', 'w', encoding='utf-8')
train_f = open('train_origin.kr', 'w', encoding='utf-8')
all_f = open('all.kr', 'r', encoding='utf-8')
all_tok_f = open('all_tok.kr', 'r', encoding='utf-8')
sen_dict = {}
for sen, sen_tok in zip(all_f.read().splitlines(), all_tok_f.read().splitlines()):
    sen_dict[sen_tok] = sen
for dev in dev_tok_f.read().splitlines():
    dev_f.write(sen_dict[dev] + '\n')
for test in test_tok_f.read().splitlines():
    test_f.write(sen_dict[test] + '\n')
for train in train_tok_f.read().splitlines():
    train_f.write(sen_dict[train] + '\n')
dev_tok_f.close()
test_tok_f.close()
train_tok_f.close()
dev_f.close()
test_f.close()
train_f.close()
all_f.close()
all_tok_f.close()

