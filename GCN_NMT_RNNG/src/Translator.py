import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Vocabulary import Vocabulary
from Models import NMT_RNNG
import re
import random
import math
import utils
import pickle

from torchviz import make_dot, make_dot_from_trace

class Data(object):
    def __init__(self):
        self.src = []
        self.tgt = []
        self.action = []
        self.deprel = []
        self.trans = []     # output of decoder


class Translator(object):
    def __init__(self,
                 srcTrain_tagged,
                 tgtTrain_tagged,
                 srcDev_tagged,
                 tgtDev_tagged,
                 srcVocaThreshold,
                 tgtVocaThreshold,
                 deprelLabelThreshold):
        #train_size = 20000
        print('Parsing target file into plain sentences & actions...')
        tgtTrain, actTrain = self.conll_to_action(tgtTrain_tagged)
        tgtDev, actDev = self.conll_to_action(tgtDev_tagged)
        print('Parsing source file into plain sentences & dependency relations...')
        srcTrain, deprelTrain = self.conll_to_deprels(srcTrain_tagged)
        srcDev, deprelDev = self.conll_to_deprels(srcDev_tagged)

        self.sourceVoc = Vocabulary(srcTrain, srcVocaThreshold, 'lang')
        self.targetVoc = Vocabulary(tgtTrain, tgtVocaThreshold, 'lang')
        self.actionVoc = Vocabulary(actTrain, None, 'action')
        self.deprelVoc = Vocabulary(deprelTrain, deprelLabelThreshold, 'deprel')
        self.trainData = []
        self.devData = []
        self.trainData = self.loadCorpus(srcTrain, tgtTrain, actTrain, deprelTrain,  self.trainData)
        self.devData = self.loadCorpus(srcDev, tgtDev, actDev, deprelDev, self.devData)

    def train(self, criterion, NLL, optimizer, train=True):
        permutation = list(range(0, len(self.trainData)))
        random.shuffle(permutation)
        batchNumber = int(math.ceil(len(self.trainData) / self.miniBatchSize))
        for batch_i in range(1, batchNumber + 1):
            print('Progress: ' + str(batch_i) + '/' + str(batchNumber) + ' mini batches')
            startIdx = (batch_i - 1) * self.miniBatchSize
            endIdx = startIdx + self.miniBatchSize
            if endIdx > len(self.trainData):
                endIdx = len(self.trainData)
            indices = permutation[startIdx:endIdx]
            batch_trainData = [self.trainData[i] for i in indices]

            loss = 0
            optimizer.zero_grad()
            index = 0
            for data_in_batch in batch_trainData:
                index += 1

                data_in_batch.src.pop()
                data_in_batch.tgt.pop()
                #remove eos

                # print(self.targetVoc.tokenList[data_in_batch.tgt[-1]][0])

                train_src = torch.LongTensor(data_in_batch.src)
                train_tgt = torch.LongTensor(data_in_batch.tgt)
                train_action = torch.LongTensor(data_in_batch.action)

                src_length = len(data_in_batch.src)
                enc_hidden = self.model.enc_init_hidden()
                uts, s_tildes = self.model(train_src, train_tgt, train_action, src_length, enc_hidden)

                predicted_words = F.log_softmax(s_tildes.view(-1, len(self.targetVoc.tokenList)), dim=1)
                torch.set_printoptions(threshold=10000)
                # print(predicted_words[0])
                print("in batch "+str(batch_i)+", "+str(index)+"th data")
                print("source: ", end="")
                for i in data_in_batch.src:
                    print(self.sourceVoc.tokenList[i][0], end=" ")
                print("\ngold: ", end="")
                for i in data_in_batch.tgt:
                    print(self.targetVoc.tokenList[i][0], end=" ")
                print("\ntarget: ", end="")
                for i in range(list(predicted_words.shape)[0]):
                    topv, topi = predicted_words[i].topk(1)
                    print(self.targetVoc.tokenList[topi][0], end=" ")
                print("\n")

                loss_t = 0
                word_cnt = 0
                for i in range(min(list(predicted_words.shape)[0], len(train_tgt))):
                    topv, topi = predicted_words[i].topk(1)
                    if self.targetVoc.tokenList[topi][0] == '.':
                        continue
                    loss_t += NLL(predicted_words[i].view(1, -1), torch.LongTensor([train_tgt[i]]))
                    word_cnt+=1
                if word_cnt != 0:
                    loss += loss_t / word_cnt

                # Backward(Action)
                loss += criterion(uts.view(-1, len(self.actionVoc.tokenList)), train_action)
            # dot = make_dot(uts, params=dict(self.model.named_parameters()))
            # with open("uts.dot", "w") as f:
            #     f.write(str(dot))
            # dot = make_dot(s_tildes, params=dict(self.model.named_parameters()))
            # with open("stildes.dot", "w") as f:
            #     f.write(str(dot))
            print("loss: ", round(loss.item(), 2))
            loss.backward()
            optimizer.step()

        return

    def loadCorpus(self, src, tgt, act, deprel, data):
        with open(src, encoding="utf-8") as f:
            for line in f:
                data.append(Data())
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '']
                for token in tokens:
                    if token in self.sourceVoc.tokenIndex:
                        data[-1].src.append(self.sourceVoc.tokenIndex[token])
                    else:
                        data[-1].src.append(self.sourceVoc.unkIndex)
                data[-1].src.append(self.sourceVoc.eosIndex)
        idx = 0
        with open(tgt, encoding="utf-8") as f:
            for line in f:
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '']
                for token in tokens:
                    if token in self.targetVoc.tokenIndex:
                        data[idx].tgt.append(self.targetVoc.tokenIndex[token])
                    else:
                        data[idx].tgt.append(self.targetVoc.unkIndex)
                data[idx].tgt.append(self.targetVoc.eosIndex)
                idx += 1
        idx = 0
        with open(act) as f:
            for line in f:
                tokens = re.split('[ \t\n]', line)
                tokens = [x for x in tokens if x != '']
                if len(tokens) > 0:
                    if tokens[0] in self.actionVoc.tokenIndex:
                        data[idx].action.append(self.actionVoc.tokenIndex[tokens[0]])
                    else:
                        if "LEFT" in tokens[0]:
                            data[idx].action.append(self.actionVoc.tokenIndex['REDUCE-LEFT-ARC(unk)'])
                        elif "RIGHT" in tokens[0]:
                            data[idx].action.append(self.actionVoc.tokenIndex['REDUCE-RIGHT-ARC(unk)'])
                        else:
                            print("Error: Unknown word except shift/reduce.")
                else:
                    idx += 1
        idx = 0
        deprel_list = pickle.load(open(deprel, 'rb'))
        for dep_sen in deprel_list:
            for dep_word in dep_sen:
                label = dep_word[0]
                if label in self.deprelVoc.tokenIndex:
                    data[idx].deprel.append((self.deprelVoc.tokenIndex[label], dep_word[1], dep_word[2]))
                else:
                    data[idx].deprel.append((self.deprelVoc.unkIndex, dep_word[1], dep_word[2]))
            idx += 1
        return data

    def conll_to_action(self, tgt):
        cnt = 0
        if 'dev' in tgt:
            oracle_fname = '../data/processed/dev.oracle.en'
            txt_fname = '../data/processed/dev.en'
        elif 'test' in tgt:
            oracle_fname = '../data/processed/test.oracle.en'
            txt_fname = '../data/processed/test.en'
        elif 'train' in tgt:
            oracle_fname = '../data/processed/train.oracle.en'
            txt_fname = '../data/processed/train.en'
        else:
            print('Error: invalid file name of ' + tgt)
            exit(1)
        oracle_f = open(oracle_fname, 'w', encoding='utf-8')
        plain_f = open(txt_fname, 'w', encoding='utf-8')
        tagged_file = open(tgt, 'r', encoding='utf-8')
        bulk = tagged_file.read()
        blocks = re.compile(r"\n{2,}").split(bulk)
        blocks = list(filter(None, blocks))
        for block in blocks:
            tokens = []
            buffer = []
            child_to_head_dict = {}
            for line in block.splitlines():
                attr_list = line.split('\t')
                tokens.append(attr_list[1])
                num = int(attr_list[0])
                head = int(attr_list[6])
                label = attr_list[7]
                node = utils.Node(num, head, label)
                child_to_head_dict[num] = head
                buffer.append(node)
            arcs = utils.write_oracle(buffer, child_to_head_dict)
            for i, token in enumerate(tokens):
                token_lowered = token.lower()
                if i == 0:
                    plain_f.write(token_lowered)
                else:
                    plain_f.write(' ')
                    plain_f.write(token_lowered)
            plain_f.write('\n')
            for arc in arcs:
                oracle_f.write(arc + '\n')
            oracle_f.write('\n')
            cnt += 1
            if cnt > 5000:
                break
        tagged_file.close()
        oracle_f.close()
        plain_f.close()
        return txt_fname, oracle_fname

    def conll_to_deprels(self, src):
        cnt = 0
        if 'dev' in src:
            deprels_fname = '../data/processed/dev.deprel.kr'
            txt_fname = '../data/processed/dev.kr'
        elif 'test' in src:
            deprels_fname = '../data/processed/test.deprel.kr'
            txt_fname = '../data/processed/test.kr'
        elif 'train' in src:
            deprels_fname = '../data/processed/train.deprel.kr'
            txt_fname = '../data/processed/train.kr'
        else:
            print('Error: invalid file name of ' + src)
            exit(1)
        deprels_f = open(deprels_fname, 'wb')
        plain_f = open(txt_fname, 'w', encoding='utf-8')
        tagged_file = open(src, 'r', encoding='utf-8')
        bulk = tagged_file.read()
        blocks = re.compile(r"\n{2,}").split(bulk)
        blocks = list(filter(None, blocks))
        deprels = []
        for block in blocks:
            deprel = []
            tokens = []
            for line in block.splitlines():
                attr_list = line.split('\t')
                tokens.append(attr_list[1])
                num = int(attr_list[0])
                head = int(attr_list[6])
                label = attr_list[7]
                deprel.append((label, head, num))
            deprels.append(deprel)
            for i, token in enumerate(tokens):
                if i == 0:
                    plain_f.write(token)
                else:
                    plain_f.write(' ')
                    plain_f.write(token)
            plain_f.write('\n')
            cnt += 1
            if cnt > 5000:
                break
        pickle.dump(deprels, deprels_f)
        tagged_file.close()
        deprels_f.close()
        plain_f.close()
        return txt_fname, deprels_fname

    def demo(self,
             inputDim,
             inputActDim,
             hiddenDim,
             hiddenEncDim,
             hiddenActDim,
             scale,
             clipThreshold,
             beamSize,
             maxLen,
             miniBatchSize,
             threadNum,
             learningRate,
             saveDirName,
             loadModelName,
             loadGradName,
             startIter,
             epochs,
             useGCN):
        self.miniBatchSize = miniBatchSize
        self.model = NMT_RNNG(self.sourceVoc,
                              self.targetVoc,
                              self.actionVoc,
                              self.deprelVoc,
                              self.trainData,
                              self.devData,
                              inputDim,
                              inputActDim,
                              hiddenEncDim,
                              hiddenDim,
                              hiddenActDim,
                              scale,
                              clipThreshold,
                              beamSize,
                              maxLen,
                              self.miniBatchSize,
                              threadNum,
                              learningRate,
                              False,
                              0,
                              saveDirName,
                              useGCN)

        translation = []    # 결과, 나중에 devData와 같은 길이의 list가 됨.
        optimizer = optim.Adam(self.model.parameters(), lr=learningRate, weight_decay=0.005, amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        NLL = nn.NLLLoss()
        print("# of Training Data:\t" + str(len(self.trainData)))
        print("# of Development Data:\t" + str(len(self.devData)))
        print("Source voc size: " + str(len(self.sourceVoc.tokenList)))
        print("Target voc size: " + str(len(self.targetVoc.tokenList)))
        print("Action voc size: " + str(len(self.actionVoc.tokenList)))
        print("Dependency Label voc size: " + str(len(self.deprelVoc.tokenList)))

        for i in range(epochs):
            print("Epoch " + str(i+1) + ' (lr = ' + str(self.model.learningRate) + ')')
            status = self.train(criterion, NLL, optimizer)

























