import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Vocabulary import Vocabulary
from Models import NMT_RNNG
import re
import random
import math


class Data(object):
    def __init__(self):
        self.src = []
        self.tgt = []
        self.action = []
        self.trans = [] # output of decoder


class Translator(object):
    def __init__(self,
                 srcTrain,
                 tgtTrain,
                 actTrain,
                 srcDev,
                 tgtDev,
                 actDev,
                 srcVocaThreshold,
                 tgtVocaThreshold):

        self.sourceVoc = Vocabulary(srcTrain, srcVocaThreshold, 'lang')
        self.targetVoc = Vocabulary(tgtTrain, tgtVocaThreshold, 'lang')
        self.actionVoc = Vocabulary(actTrain, None, 'action')
        self.trainData = []
        self.devData = []
        self.trainData = self.loadCorpus(srcTrain, tgtTrain, actTrain, self.trainData)
        self.devData = self.loadCorpus(srcDev, tgtDev, actDev, self.devData)

    def train(self, criterion, NLL, optimizer, train=True):
        permutation = list(range(0, len(self.trainData)))
        random.shuffle(permutation)
        batchNumber = int(math.ceil(len(self.trainData) / self.miniBatchSize))
        for i in range(1, batchNumber + 1):
            print('Progress: ' + str(i) + '/' + str(batchNumber) + ' mini batches')
            startIdx = (i - 1) * self.miniBatchSize
            endIdx = startIdx + self.miniBatchSize
            if endIdx > len(self.trainData):
                endIdx = len(self.trainData)
            indices = permutation[startIdx:endIdx]
            batch_trainData = [self.trainData[i] for i in indices]

            for data_in_batch in batch_trainData:
                optimizer.zero_grad()
                train_src = torch.LongTensor(data_in_batch.src)
                train_tgt = torch.LongTensor(data_in_batch.tgt)
                train_action = torch.LongTensor(data_in_batch.action)

                src_length = len(data_in_batch.src)
                enc_hidden = self.model.enc_init_hidden()
                uts, s_tildes = self.model(train_src, train_tgt, train_action, src_length, enc_hidden, criterion)
                #print(s_tildes)
                # Backward(Action)
                act_loss = criterion(uts.view(-1, len(self.actionVoc.tokenList)), train_action)
                act_loss.backward(retain_graph=True)

                predicted_words = F.softmax(s_tildes.view(-1, len(self.targetVoc.tokenList)), dim=1)
                #print(predicted_words)
                loss = 0
                for i in range(len(train_tgt)):
                    # print(i, train_tgt[i])
                    # print(predicted_words[i][train_tgt[i]])
                    # print(-torch.log(predicted_words[i][train_tgt[i]]))
                    try:
                        topv, topi = predicted_words[i].topk(1)
                        print(self.targetVoc.tokenList[topi], end=" ")
                        loss += -torch.log(predicted_words[i][train_tgt[i]])
                    except:
                        break
                print("")
                # loss = NLL(predicted_words, train_tgt[:len(predicted_words)])
                loss.backward()
                optimizer.step()
                print("loss: ", loss, end="\t")
                print("act_loss:", act_loss)
        return

    def loadCorpus(self, src, tgt, act, data):
        with open(src) as f:
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
                if tokens:
                    if tokens[0] in self.actionVoc.tokenIndex:
                        data[idx].action.append(self.actionVoc.tokenIndex[tokens[0]])
                    else:
                        print("Error: Unknown word except shift/reduce.")
                        exit(1)
                else:
                    idx += 1
        return data

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
             epochs):
        self.miniBatchSize = miniBatchSize
        self.model = NMT_RNNG(self.sourceVoc,
                              self.targetVoc,
                              self.actionVoc,
                              self.trainData,
                              self.devData,
                              inputDim,
                              inputActDim,
                              hiddenDim,
                              hiddenEncDim,
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
                              saveDirName)

        translation = []    # 결과, 나중에 devData와 같은 길이의 list가 됨.
        optimizer = optim.SGD(self.model.parameters(), lr=learningRate)
        criterion = nn.CrossEntropyLoss()
        NLL = nn.NLLLoss()
        print("# of Training Data:\t" + str(len(self.trainData)))
        print("# of Development Data:\t" + str(len(self.devData)))
        print("Source voc size: " + str(len(self.sourceVoc.tokenList)))
        print("Target voc size: " + str(len(self.targetVoc.tokenList)))
        print("Action voc size: " + str(len(self.actionVoc.tokenList)))

        for i in range(epochs):
            print("Epoch " + str(i+1) + ' (lr = ' + str(self.model.learningRate) + ')')
            status = self.train(criterion, NLL, optimizer)