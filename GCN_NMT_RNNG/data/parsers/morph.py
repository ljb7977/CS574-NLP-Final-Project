from optparse import OptionParser
from konlpy.tag import Komoran

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# global variable
VERBOSE = 0
test_kr = open('./test.kr', 'r')
test_tokenized = open('./test_tok.kr', 'w')
test_parsed_kr = open('./test_tok_conll.kr', 'w')
train_kr = open('../en-kr/train.kr', 'r')
train_tokenized = open('../en-kr/train_tok.kr', 'w')
train_parsed_kr = open('../en-kr/train_tok_conll.kr', 'w')
dev_kr = open('../en-kr/dev.kr', 'r')
dev_tokenized = open('../en-kr/dev_tok.kr', 'w')
dev_parsed_kr = open('../en-kr/dev_tok_conll.kr', 'w')
parser = OptionParser()
parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
(options, args) = parser.parse_args()
if options.verbose : VERBOSE = 1
komoran = Komoran()
for line in test_kr.read().splitlines():
	tok_line = ''
	analyzed = komoran.pos(line)
	seq = 1
	cnt = 0
	for morph, tag in analyzed:
		if cnt == 0:
			cnt += 1
		else:
			tok_line += ' '
		tok_line += morph
		tp = str(seq) + '\t' + morph + '\t' + morph + '\t' + tag + '\t' + tag + '\t' + '_' + '\t' + str(0) + '\t' + '_' + '\t' + '_' + '\t' + '_'
		print(tp)
		test_parsed_kr.write(tp + '\n')
		seq += 1
	test_tokenized.write(tok_line)
	test_tokenized.write('\n')
	test_parsed_kr.write('\n')
test_kr.close()
test_parsed_kr.close()
for line in dev_kr.read().splitlines():
	tok_line = ''
	cnt = 0
	analyzed = komoran.pos(line)
	seq = 1
	for morph, tag in analyzed:
		if cnt == 0:
			cnt += 1
		else:
			tok_line += ' '
		tok_line += morph
		tp = str(seq) + '\t' + morph + '\t' + morph + '\t' + tag + '\t' + tag + '\t' + '_' + '\t' + str(0) + '\t' + '_' + '\t' + '_' + '\t' + '_'
		print(tp)
		dev_parsed_kr.write(tp + '\n')
		seq += 1
	dev_tokenized.write(tok_line)
	dev_tokenized.write('\n')
	dev_parsed_kr.write('\n')
dev_kr.close()
dev_parsed_kr.close()
for line in train_kr.read().splitlines():
	analyzed = komoran.pos(line)
	seq = 1
	tok_line = ''
	cnt = 0
	for morph, tag in analyzed:
		if cnt == 0:
			cnt += 1
		else:
			tok_line += ' '
		tok_line += morph
		tp = str(seq) + '\t' + morph + '\t' + morph + '\t' + tag + '\t' + tag + '\t' + '_' + '\t' + str(0) + '\t' + '_' + '\t' + '_' + '\t' + '_'
		print(tp)
		train_parsed_kr.write(tp + '\n')
		seq += 1
	train_tokenized.write(tok_line)
	train_tokenized.write('\n')
	train_parsed_kr.write('\n')
train_kr.close()
train_parsed_kr.close()
test_tokenized.close()
dev_tokenized.close()
train_tokenized.close()



