import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import utils, copy


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

        # TODO init
        self.stack = []
        self.headList = []
        self.embedList = []
        self.actState = []

        self.del_embedVec = {}
        self.headStack = []
        self.headList = []
        self.embedStack = []
        self.embedList = []

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
        self.decInitAffine = nn.Linear(self.hiddenEncDim * 2, self.hiddenDim)
        utils.linear_init(self.decInitAffine, self.scale)
        self.actInitAffine = nn.Linear(self.hiddenEncDim * 2, self.hiddenActDim)
        utils.linear_init(self.actInitAffine, self.scale)
        self.outBufInitAffine = nn.Linear(self.hiddenEncDim * 2, self.hiddenDim)
        utils.linear_init(self.outBufInitAffine, self.scale)

        self.utAffine = nn.Linear(self.hiddenDim * 2 + self.hiddenActDim, self.hiddenDim)
        utils.linear_init(self.utAffine, self.scale)
        self.stildeAffine = nn.Linear(self.hiddenDim + self.hiddenActDim * 2, self.hiddenDim)
        utils.linear_init(self.stildeAffine, self.scale)
        self.embedVecAffine = nn.Linear(self.inputDim * 2 + self.inputActDim, self.inputDim)
        utils.linear_init(self.embedVecAffine, self.scale)

        # embedding matrices는 보통 inputDim*len(Voc)형태로 만들어져야하는데, 일단 보류
        self.sourceEmbed = torch.Tensor(self.inputDim, len(self.sourceVoc.tokenList))
        init.uniform_(self.sourceEmbed, 0., self.scale)
        self.targetEmbed = torch.Tensor(self.inputDim, len(self.targetVoc.tokenList))
        init.uniform_(self.targetEmbed, 0., self.scale)
        self.actionEmbed = torch.Tensor(self.inputActDim, len(self.actionVoc.tokenList))
        init.uniform_(self.actionEmbed, 0., self.scale)

        self.Wgeneral = torch.Tensor(self.hiddenDim, hiddenEncDim * 2)
        init.uniform_(self.Wgeneral, 0., self.scale)

        self.zeros = torch.zeros(self.hiddenDim)
        self.zerosEnc = torch.zeros(self.hiddenEncDim)
        self.zeros2 = torch.zeros(self.hiddenEncDim * 2)
        self.zerosAct = torch.zeros(self.hiddenActDim)

        # for automatic tuning
        self.prevPerp = float('inf')

    def decoderAction(self, actNum, i, train):
        # TODO call forward for action LSTM?
        if train:
            self.actState[i].delc = torch.zeros(self.hiddenActDim)
            self.actState[i].delh = torch.zeros(self.hiddenActDim)
        return

    def compositionFunc(self, phraseNum, head, dependent, relation):
        embedVecEnd = torch.cat((head, dependent, relation), 1)
        self.embedVecEnd[phraseNum - self.tgtLen] = embedVecEnd
        embed = self.embedVecAffine(embedVecEnd)

        return F.tanh(embed)

    def decoderReduceLeft(self, data, phraseNum, actNum, k, train):
        top = self.reduceHeadStack(k)
        rightNum, leftNum = self.reduceStack()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)

        if rightNum < self.tgtLen and leftNum < self.tgtLen: # word embedding & word embeddding
            # head = self.targetEmbed.col(data.tgt[rightNum])
            # dependent = self.targetEmbed.col(data.tgt[leftNum])
            # relation = self.self.actionEmbed.col(data.action[actNum])
            head = torch.t(torch.index_select(self.targetEmbed, 1, [data.tgt[rightNum]]))  # parent: right
            dependent = torch.t(torch.index_select(self.targetEmbed, 1, [data.tgt[leftNum]])) # child: left
            # self.compositionFunc(self.embedVec[phraseNum - self.tgtLen],
            #                      self.targetEmbed.col(data.tgt[rightNum]),
            #                      self.targetEmbed.col(data.tgt[leftNum]),
            #                      self.actionEmbed.col(data.action[actNum]),
            #                      self.embedVecEnd[phraseNum - self.tgtLen])
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen

            head = self.embedVec[rightNum]                                                      # parent: right
            dependent = torch.t(torch.index_select(self.targetEmbed, 1, [data.tgt[leftNum]]))   # child: left,
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen

            head = torch.t(torch.index_select(self.targetEmbed, 1, [data.tgt[rightNum]]))       # parent: right
            dependent = self.embedVec[leftNum]                                                  # child: left
            #
            # self.compositionFunc(,
            #                      self.targetEmbed.col(data.tgt[rightNum]),  # parent: right
            #                      self.embedVec[leftNum],                     # child: left,
            #                      self.actionEmbed.col(data.action[actNum]),
            #                      self.embedVecEnd[phraseNum - self.tgtLen])
        else:
            rightNum -= self.tgtLen
            leftNum -= self.tgtLen

            head = self.embedVect[rightNum]         # parent: right
            dependent = self.embedVec[leftNum]      # child: left

            # self.compositionFunc(arg.embedVec[phraseNum - arg.tgtLen],
            #                      arg.embedVec[rightNum],  # parent: right
            #                      arg.embedVec[leftNum],  # child: left,
            #                      self.actionEmbed.col(data.action[actNum]),
            #                      arg.embedVecEnd[phraseNum - arg.tgtLen])

        relation = torch.t(torch.index_select(self.actionEmbed, 1, [data.action[actNum]]))
        self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(phraseNum, head, dependent, relation)

        self.outBuf(self.embedVec[phraseNum - self.tgtLen]) #TODO LSTM forward

        # self.outBuf.forward(arg.embedVec[phraseNum - arg.tgtLen],
        #                      arg.outBufState[top], arg.outBufState[k]); #(xt, prev, cur)
        # TODO: not needed?
        if train:
            self.outBufState[k].delc = copy.deepcopy(self.zeros)
            self.outBufState[k].delh = copy.deepcopy(self.zeros)
        self.embedStack.append(phraseNum)

    def decoderReduceRight(self, data, phraseNum, actNum, k, train):
        top = self.reduceHeadStack(k)
        rightNum, leftNum = self.reduceStack()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)


        if rightNum < self.tgtLen and leftNum < self.tgtLen:
            # word embedding & word embeddding
            head = torch.t(torch.index_select(self.targetEmbed, 1, [data.tgt[leftNum]]))
            dependent = torch.t(torch.index_select(self.targetEmbed, 1, [data.tgt[rightNum]]))
            relation = torch.t(torch.index_select(self.actionEmbed, 1, [data.action[actNum]]))

            self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(phraseNum, head, dependent, relation)
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen
            head = torch.t(torch.index_select(self.targetEmbed, 1, [data.tgt[leftNum]])) # parent: left
            dependent = self.embedVec[rightNum] # child: right
            relation = torch.t(torch.index_select(self.actionEmbed, 1, [data.action[actNum]]))

            self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(phraseNum, head, dependent, relation)
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen
            self.compositionFunc(self.embedVec[phraseNum - self.tgtLen],
                                 self.embedVec[leftNum],                     # parent: left
                                 self.targetEmbed.col(data.tgt[rightNum]),  # child: right
                                 self.actionEmbed.col(data.action[actNum]),
                                 self.embedVecEnd[phraseNum - self.tgtLen])
        else:
            rightNum -= self.tgtLen
            leftNum -= self.tgtLen
            self.compositionFunc(self.embedVec[phraseNum - self.tgtLen],
                                 self.embedVec[leftNum],  # parent: left,
                                 self.embedVec[rightNum],  # child: right
                                 self.actionEmbed.col(data.action[actNum]),
                                 self.embedVecEnd[phraseNum - self.tgtLen])

        # self.outBuf.forward(self.embedVec[phraseNum - self.tgtLen],
        #                      self.outBufState[top], self.outBufState[k]); #(xt, prev, cur)
        # TODO: not needed?
        if train:
            self.outBufState[k].delc = copy.deepcopy(self.zeros)
            self.outBufState[k].delh = copy.deepcopy(self.zeros)
        self.embedStack.append(phraseNum)
        return

    def reduceHeadStack(self, k):
        self.stack.pop()
        self.stack.pop()
        top = self.stack.pop()
        self.stack.append(k)
        return top

    def reduceStack(self):
        right = self.stack.pop()
        self.stack.pop()
        left = self.stack.pop()
        self.stack.pop()
        return right, left

    def decoderAttention(self, arg, decState, contextSeq, s_tilde, stildeEnd):
        contextSeq = torch.zeros(self.hiddenEncDim * 2)
        self.calculateAlpha(arg, decState)
        for j in range(arg.srcLen):
            contextSeq += arg.alphaSeqVec.coeff(j, 0) * arg.biEncState[j]

        stildeEnd.segment(0, this.hiddenDim).noalias() = decState.h
        stildeEnd.segment(self.hiddenDim, self.hiddenEncDim * 2).noalias() = contextSeq

        self.stildeAffine.forward(stildeEnd, s_tilde)

    def decoderAttention(self, arg, i, train):
        arg.contextSeqList[i] = torch.zeros(self.hiddenEncDim * 2)

        self.calculateAlpha(arg, arg.decState[i], i)

        for j in range(arg.srcLen):
            arg.contextSeqList[i] += arg.alphaSeq.coeff(j, i) * arg.biEncState[j]

        arg.stildeEnd[i].segment(0, self.hiddenDim).noalias() = arg.decState[i].h
        arg.stildeEnd[i].segment(self.hiddenDim, self.hiddenEncDim * 2) = arg.contextSeqList[i]

        self.stildeAffine.forward(arg.stildeEnd[i], arg.s_tilde[i]);

    def calculateAlpha(self, arg, decState):
        for i in range(arg.srcLen):
            arg.alphaSeqVec.coeffRef(i, 0) = decState.h.dot(self.Wgeneral * arg.biEncState[i])

        # softmax of ``alphaSeq``
        arg.alphaSeqVec.array() -= arg.alphaSeqVec.maxCoeff() # stable softmax
        arg.alphaSeqVec = arg.alphaSeqVec.array().exp() #exp() operation for all elements; np.exp(alphaSeq)
        arg.alphaSeqVec /= arg.alphaSeqVec.array().sum() # alphaSeq.sum()

    def calculateAlpha(self, arg, decState, colNum):
        for i in range(srcLen):
            arg.alphaSeq.coeffRef(i, colNum) = decState.h.dot(self.Wgeneral*arg.biEncState[i])
        arg.alphaSeq.col(colNum).array() -= arg.alphaSeq.col(colNum).maxCoeff() #stable softmax
        arg.alphaSeq.col(colNum) = arg.alphaSeq.col(colNum).array().exp() #exp()  operation for all elements; np.exp(alphaSeq)
        arg.alphaSeq.col(colNum) /= arg.alphaSeq.col(colNum).array().sum(); #alphaSeq.sum()

    def calcLoss(self, data, train):
        loss = 0.0
        lossAct = 0.0
        j=0, k=0
        phraseNum = data.tgt.size()
        actNum = None

        self.del_embedVec = {}
        self.headStack = []
        self.headList = {}
        self.embedStack = []
        self.embedList = {}
        # this->biEncode(data, arg, false)

        # k == 0
        # self.outBufInitAffine.forward(arg.encStateEnd, arg.outBufState[k]->h);
        arg.outBufState[k].c = torch.zeros(self.hiddenDim)

        arg.headStack.append(k)
        k+=1

        for i in range(arg.actLen): #SoftmaxAct calculation
            actNum = data.action[i]
            self.decoderAction(arg, arg.actState, data.action[i - 1], i, False) # PUSH
            if self.actionVoc.tokenList[actNum].action == 0: # 0: Shift
                arg.headStack.append(k)
                #1) Let a decoder proceed one step; PUSH
                self.decoder(arg, arg.decState, arg.s_tilde[j - 1], data.tgt[j - 1], j, False)

                # Attention
                self.decoderAttention(arg, arg.decState[j], arg.contextSeqList[j], arg.s_tilde[j], arg.stildeEnd[j])
                if not self.useBlackOut:
                    self.softmax.calcDist(arg.s_tilde[j], arg.targetDist)
                    loss += self.softmax.calcLoss(arg.targetDist, data.tgt[j])
                else:
                    if train: #word prediction
                        arg.blackOutState[0].sample[0] = data.tgt[j];
                        arg.blackOutState[0].weight.col(0) = self.blackOut.weight.col(data.tgt[j])
                        arg.blackOutState[0].bias.coeffRef(0, 0) = self.blackOut.bias.coeff(data.tgt[j], 0)

                        self.blackOut.calcSampledDist2(arg.s_tilde[j], arg.targetDist, arg.blackOutState[0])
                        loss += self.blackOut.calcSampledLoss(arg.targetDist) #Softmax
                    else: # Test Time
                        self.blackOut.calcDist(arg.s_tilde[j], arg.targetDist) #Softmax
                        loss += self.blackOut.calcLoss(arg.targetDist, data.tgt[j]) #Softmax
                # 2) Let the output buffer proceed one step, though the computed unit is not used at this step; PUSH
                self.outBuf.forward(self.targetEmbed.col(data.tgt[j]),
                                    arg.outBufState[k - 1], arg.outBufState[k])

                arg.embedStack.append(j)

                # SoftmaxAct calculation(o: output buffer, s: stack, and h: action)
                arg.utEnd[0].segment(0, self.hiddenDim) = arg.decState[j].h
                j+=1
            elif self.actionVoc.tokenList[actNum].action == 1: # 1: Reduce - Left
                self.decoderReduceLeft(data, arg, phraseNum, i - 1, k, False)
                phraseNum+=1

                # SoftmaxAct calculation(o: output buffer, s: stack, and h: action)
                arg.utEnd[0].segment(0, self.hiddenDim) = arg.decState[j - 1].h;

            elif self.actionVoc.tokenList[actNum].action == 2: # 2: Reduce - Right
                self.decoderReduceRight(data, arg, phraseNum, i - 1, k, False)
                phraseNum+=1

                # SoftmaxAct calculation(o: output buffer, s: stack, and h: action)
                arg.utEnd[0].segment(0, this->hiddenDim) = arg.decState[j - 1].h

            else:
                raise("Error Non-Shift/Reduce")

            arg.utEnd[0].segment(self.hiddenDim, self.hiddenDim) = arg.outBufState[k - 1].h
            arg.utEnd[0].segment(self.hiddenDim * 2, self.hiddenActDim) = arg.actState[i].h
            self.utAffine.forward(arg.utEnd[0], arg.ut[0])

            self.softmaxAct.calcDist(arg.ut[0], arg.actionDist)
            lossAct += self.softmaxAct.calcLoss(arg.actionDist, data.action[i])

            k+=1

        arg.clear()
        return [loss, lossAct]

class Arg:
    def __init__(self, data, train):
        self.srcLen = data.src.size()
        self.tgtLen = data.tgt.size()
        self.actLen = data.action.size()
        self.outBufState = []
        self.headStack = []
        self.del_embedVec = {}
        self.headList = {}
        self.embedStack = []
        self.embedList = {}

        if train:
            self.alphaSeq = torch.zeros(self.srcLen, self.tgtLen)
            self.encStateEnd = torch.zeros(self.hiddenEncDim * 2)
            self.del_encStateEnd = torch.zeros(self.hiddenEncDim * 2)
            self.del_alphaSeq = torch.zeros(self.srcLen)
            self.del_alphaSeqTmp = torch.zeros(self.hiddenEncDim)
            self.del_WgeneralTmp = torch.zeros(self.hiddenEncDim * 2)
            self.alphaSeqVec = torch.zeros(self.srcLen)
        else:
            self.alphaSeqVec = torch.zeros(self.srcLen)

    def clear(self):
        self.del_embedVec = {}
        self.headStack = []
        self.headList = {}
        self.embedStack = []
        self.embedList = {}

class Data:
    def __init__(self, _train_src, _train_tgt, _train_action):
        self.train_src = _train_src
        self.train_tgt = _train_tgt
        self.train_action = _train_action
