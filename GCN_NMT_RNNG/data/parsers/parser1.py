from xml.etree import ElementTree as ET
import os
import re

# make all.en and all.kr

xml_cnt = 0     # 506197
all_cnt = 0     # 409093

f_kr = open('all.kr', 'w', encoding='utf-8')
with open('../en-kr.xml', 'r', encoding='utf-8') as file:
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
        if en == '' or kr == '' or en == None or kr == None or len(en) > 80 or '+' in en or '+' in kr or '@' in en or '@' in kr or '#' in kr or '#' in en or '%' in kr or '%' in en or '&' in kr or '&' in en or '*' in kr or '*' in en or '$' in kr or '$' in en or '-' in kr or '-' in en or '=' in kr or '=' in en or '(' in kr or ')' in en or ')' in kr or '(' in en or '^' in kr or '^' in en or '{' in kr or '}' in en:
            continue
        else:
            f_kr.write(kr + '\n')
            all_cnt += 1
f_kr.close()

print(xml_cnt)
print(all_cnt)












