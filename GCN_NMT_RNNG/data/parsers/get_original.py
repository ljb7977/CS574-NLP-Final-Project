from optparse import OptionParser
from konlpy.tag import Komoran

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# global variable
VERBOSE = 0
kr = open('./all.kr', 'r')
tokenized = open('./all_tok.kr', 'w')
parser = OptionParser()
parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
(options, args) = parser.parse_args()
if options.verbose : VERBOSE = 1
komoran = Komoran()
for line in kr.read().splitlines():
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
        seq += 1
    tokenized.write(tok_line)
    tokenized.write('\n')
kr.close()
tokenized.close()