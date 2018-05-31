import torch
from Vocabulary import Vocabulary


def loadCorpus(src, tgt, act):
    data = []
    with open(src) as f:
        for line in f:
            data.append([line])
    idx = 0
    with open(tgt) as f:
        for line in f:
            data[idx].append(line)
    idx = 0
    with open(act) as f:
        for line in f:
            data[idx].append(line)
    return data


def demo(srcTrain,
         tgtTrain,
         actTrain,
         srcDev,
         tgtDev,
         actDev,
         inputDim,
         inputActDim,
         hiddenEncDim,
         hiddenDim,
         hiddenAction,
         scale,
         useBlackOut,
         blackOutSampleNum,
         blackOutAlpha,
         clipTreshold,
         beamSize,
         maxLen,
         miniBatchSize,
         threadNum,
         learningRate,
         srcVocaThreshold,
         tgtVocaThreshold,
         saveDirName):
    sourceVoc = Vocabulary(srcTrain, srcVocaThreshold, 'lang')
    targetVoc = Vocabulary(tgtTrain, tgtVocaThreshold, 'lang')
    actionVoc = Vocabulary(actTrain, None, 'action')















































