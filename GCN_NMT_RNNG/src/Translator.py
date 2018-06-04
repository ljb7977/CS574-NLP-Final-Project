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
        self.trans = []


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

    def train(self):
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
        with open(tgt) as f:
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
                              'SGD',
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
        my_optimizer = optim.SGD(self.model.parameters(), lr=learningRate)
        print("# of Training Data:\t" + str(len(self.trainData)))
        print("# of Development Data:\t" + str(len(self.devData)))
        print("Source voc size: " + str(len(self.sourceVoc.tokenList)))
        print("Target voc size: " + str(len(self.targetVoc.tokenList)))
        print("Action voc size: " + str(len(self.actionVoc.tokenList)))
        print(self.sourceVoc.tokenIndex)
        for i in range(epochs):
            print("Epoch " + str(i+1) + ' (lr = ' + str(self.model.learningRate) + ')')
            status = self.train()









































