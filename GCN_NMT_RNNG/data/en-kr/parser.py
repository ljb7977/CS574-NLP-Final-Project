from xml.etree import ElementTree as ET
import os
import re

# make all.en and all.kr
'''
xml_cnt = 0     # 506197
all_cnt = 0     # 506112

f_en = open('all.en', 'w', encoding='utf-8')
f_kr = open('all.kr', 'w', encoding='utf-8')
with open('en-kr.xml', 'r', encoding='utf-8') as file:
    str_xml = file.read()
    root = ET.XML(str_xml)
    en_sentence = []
    kr_sentence = []
    for child in root:
        for c in child:
            lang = c.get('lang')
            sentence = c.text
            if lang == 'en':
                en_sentence.append(sentence)
            elif lang == 'kr':
                kr_sentence.append(sentence)
                xml_cnt += 1
    for en, kr in zip(en_sentence, kr_sentence):
        if en == '' or kr == '' or en == None or kr == None:
            continue
        else:
            f_en.write(en + '\n')
            f_kr.write(kr + '\n')
            all_cnt += 1
f_en.close()
f_kr.close()

print(xml_cnt)
print(all_cnt)
'''

# make dev|train data
import random

all_en = open('all.en', 'r', encoding='utf-8')
all_kr = open('all.kr', 'r', encoding='utf-8')

dev_en = open('dev.en', 'w', encoding='utf-8')
dev_kr = open('dev.kr', 'w', encoding='utf-8')
train_en = open('train.en', 'w', encoding='utf-8')
train_kr = open('train.kr', 'w', encoding='utf-8')

all_en_lines = all_en.read().splitlines()
all_kr_lines = all_kr.read().splitlines()

our_corpora = [(en, kr) for en, kr in zip(all_en_lines, all_kr_lines)]
random.shuffle(our_corpora)

train_cnt = 0
dev_cnt = 0
for i, (en, kr) in enumerate(our_corpora, 1):
    if i > 1000:
        train_en.write(en + '\n')
        train_kr.write(kr + '\n')
        train_cnt += 1
    else:
        dev_en.write(en + '\n')
        dev_kr.write(kr + '\n')
        dev_cnt += 1
print(train_cnt)    # 505112
print(dev_cnt)      # 1000

all_en.close()
all_kr.close()
dev_en.close()
dev_kr.close()
train_en.close()
train_kr.close()













