import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Vocabulary import Vocabulary
from Models import NMT_RNNG
import re

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

    def demo2(self,
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
              startIter):

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
                              miniBatchSize,
                              threadNum,
                              learningRate,
                              False,
                              startIter,
                              saveDirName)

        translation = []    # 결과, 나중에 devData와 같은 길이의 list가 됨.

        print("# of Training Data:\t" + str(len(self.trainData)))
        print("# of Development Data:\t" + str(len(self.devData)))
        print("Source voc size: " + str(len(self.sourceVoc.tokenList)))
        print("Target voc size: " + str(len(self.targetVoc.tokenList)))
        print("Action voc size: " + str(len(self.actionVoc.tokenList)))








































