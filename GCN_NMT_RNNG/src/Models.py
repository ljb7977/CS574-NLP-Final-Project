import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import utils


class NMT_RNNG(nn.Module):
    def __init__(self,
                 sourceVoc,
                 targetVoc,
                 actionVoc,
                 trainData,
                 devData,
                 inputDim,
                 inputActDim,
                 hiddenEncDim,
                 hiddenDim,
                 hiddenActDim,
                 scale,
                 clipThreshold,
                 beamSize,
                 maxLen,
                 miniBatchSize,
                 threadNum,
                 learningRate,
                 isTest,
                 startIter,
                 saveDirName):
        super(NMT_RNNG, self).__init__()
        self.sourceVoc = sourceVoc
        self.targetVoc = targetVoc
        self.actionVoc = actionVoc
        self.trainData = trainData
        self.devData = devData
        self.inputDim = inputDim
        self.inputActDim = inputActDim
        self.hiddenEncDim = hiddenEncDim
        self.hiddenDim = hiddenDim
        self.hiddenActDim = hiddenActDim
        self.scale = scale
        self.clipThreshold = clipThreshold
        self.beamSize = beamSize
        self.maxLen = maxLen
        self.miniBatchSize = miniBatchSize
        self.threadNum = threadNum
        self.learningRate = learningRate
        self.isTest = isTest
        self.startIter = startIter
        self.saveDirName = saveDirName

        # 여기서 모델 정의 파파팟

        # encoder
        self.enc = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenEncDim, bidirectional=True)
        utils.lstm_init_uniform_weights(self.enc, self.scale)
        # decoder
        self.dec = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenDim, bidirectional=True)
        utils.lstm_init_uniform_weights(self.dec, self.scale)
        # action
        self.act = nn.LSTM(input_size=self.inputActDim, hidden_size=self.hiddenActDim)
        utils.lstm_init_uniform_weights(self.act, self.scale)
        # out buffer
        self.outBuf = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenDim)
        utils.lstm_init_uniform_weights(self.outBuf, self.scale)

        utils.set_forget_bias(self.enc, 1.0)
        utils.set_forget_bias(self.dec, 1.0)
        utils.set_forget_bias(self.act, 1.0)
        utils.set_forget_bias(self.outBuf, 1.0)

        # affine
        # linear later들 전부 activation function이 tanh인데 이건 forward에서 해야함
        self.decInitAffine = nn.Linear(self.hiddenEncDim*2, self.hiddenDim)
        utils.linear_init(self.decInitAffine, self.scale)
        self.actInitAffine = nn.Linear(self.hiddenEncDim*2, self.hiddenActDim)
        utils.linear_init(self.actInitAffine, self.scale)
        self.outBufInitAffine = nn.Linear(self.hiddenEncDim*2, self.hiddenDim)
        utils.linear_init(self.outBufInitAffine, self.scale)

        self.utAffine = nn.Linear(self.hiddenDim*2 + self.hiddenActDim, self.hiddenDim)
        utils.linear_init(self.utAffine, self.scale)
        self.stildeAffine = nn.Linear(self.hiddenDim + self.hiddenActDim*2, self.hiddenDim)
        utils.linear_init(self.stildeAffine, self.scale)
        self.embedVecAffine = nn.Linear(self.inputDim*2 + self.inputActDim, self.inputDim)
        utils.linear_init(self.embedVecAffine, self.scale)

        # embedding matrices는 보통 inputDim*len(Voc)형태로 만들어져야하는데, 일단 보류
        self.sourceEmbed = torch.Tensor(self.inputDim, len(self.sourceVoc.tokenList))
        init.uniform_(self.sourceEmbed, 0., self.scale)
        self.targetEmbed = torch.Tensor(self.inputDim, len(self.targetVoc.tokenList))
        init.uniform_(self.targetEmbed, 0., self.scale)
        self.actionEmbed = torch.Tensor(self.inputActDim, len(self.actionVoc.tokenList))
        init.uniform_(self.actionEmbed, 0., self.scale)

        self.Wgeneral = torch.Tensor(self.hiddenDim, hiddenEncDim*2)
        init.uniform_(self.Wgeneral, 0., self.scale)

        self.zeros = torch.zeros(self.hiddenDim)
        self.zerosEnc = torch.zeros(self.hiddenEncDim)
        self.zeros2 = torch.zeros(self.hiddenEncDim*2)
        self.zerosAct = torch.zeros(self.hiddenActDim)

        # for automatic tuning
        self.prevPerp = float('inf')

    def biEncode(self, batch_trainData, train=True):

        return




































