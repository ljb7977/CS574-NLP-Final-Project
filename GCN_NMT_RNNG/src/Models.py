import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import utils, copy
from torch.autograd import Variable

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

        self.headStack = []
        self.headList = []
        self.embedStack = []
        self.embedList = []

        # 여기서 모델 정의 파파팟

        # encoder
        self.srcEmbedding = nn.Embedding(len(self.sourceVoc.tokenList), self.inputDim)
        self.enc = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenEncDim, bidirectional=True)
        utils.lstm_init_uniform_weights(self.enc, self.scale)
        utils.set_forget_bias(self.enc, 1.0)

        # decoder
        self.dec = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenDim, bidirectional=True)
        utils.lstm_init_uniform_weights(self.dec, self.scale)
        utils.set_forget_bias(self.enc, 1.0)

        # action
        self.act = StackLSTM(input_size=self.inputActDim, hidden_size=self.hiddenActDim)
        utils.lstm_init_uniform_weights(self.act, self.scale)
        utils.set_forget_bias(self.act, 1.0)
        # out buffer, RNNG stack
        self.outBuf = StackLSTM(input_size=self.inputDim, hidden_size=self.hiddenDim)
        utils.lstm_init_uniform_weights(self.outBuf, self.scale)
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
        #stilde Affine: attention

        self.embedVecAffine = nn.Linear(self.inputDim * 2 + self.inputActDim, self.inputDim)
        utils.linear_init(self.embedVecAffine, self.scale)

        # embedding matrices는 보통 inputDim*len(Voc)형태로 만들어져야하는데, 일단 보류
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

    def enc_init_hidden(self):
        weight = next(self.parameters()).data
        return (Variable(weight.new(2, 1, self.hiddenEncDim).zero_()),
                Variable(weight.new(2, 1, self.hiddenEncDim).zero_()))

    def forward(self, src, tgt, actions, src_length, enc_hidden, train=True):
        utEnds = []
        uts = []

        output, (enc_last_state, last_cell) = self.encode(src, src_length, enc_hidden)
        # 이 디코더의 output을 action sLSTM과 decoder LSTM에 넣어야 한다.
        j, k, top = 0, 0, 0
        self.headStack.append(k)
        k += 1
        length = len(src)
        phraseNum = len(tgt)

        # i == 0

        act_h1, act_c1 = self.decoderAction(enc_last_state)
        # 일단 0일때는 action 0으로 설정
        #처음 액션은 무조건 shift이므로, 그것만 처리한다.
        self.headStack.append(k)
        dec_h1, dec_c1 = self.decoder(enc_last_state)
        s_tilde = self.decoderAttention(dec_h1)
        out_h1, out_c1 = self.outBuf(tgt[j]) #0번째 target word embedding 넣음
        self.embedStack.append(j)

        utEnd = torch.cat((dec_h1, out_h1, act_h1))
        utEnds.append(utEnd)
        ut = F.tanh(self.utAffine(utEnd))
        uts.append(ut)

        j+=1
        k+=1

        for i in range(1, len(actions)):
            actNum = actions[i]
            act_h1, act_c1 = self.decoderAction(actions[i-1], act_h1, act_c1) #put prev action

            if self.getAction(actNum) == 0: #shift action
                self.headStack.append(k) #push to headStack

                dec_h1, dec_c1 = self.decoder(s_tilde, (dec_h1, dec_c1)) #TODO decoder forward 1 step with stilde
                s_tilde = self.decoderAttention(dec_h1)
                self.outBuf.push(self.col(self.targetEmbed, tgt[j]), (out_h1, out_c1)) #outbut forward
                self.embedStack.append(j)
                j+=1
            elif self.getAction(actNum) == 1: # Reduce left
                self.decoderReduceLeft(phraseNum, i-1, k, True, actions)
                phraseNum+=1
            elif self.getAction(actNum) == 2: #reduce right
                self.decoderReduceRight(phraseNum, i-1, k, True, actions)
                phraseNum+=1
            else:
                raise("Action Error: undefined Action")

            utEnd = torch.cat((dec_h1, out_h1, act_h1))
            utEnds.append(utEnd)
            ut = F.tanh(self.utAffine(utEnd))
            uts.append(ut)

            k+=1
            # self.ut.forward()
        return #act, dec, out h1 and c1, s_tilde

    def getAction(self, actNum):
        return self.actionVoc.tokenList[actNum][2]

    def col(self, tensor, i):
        return torch.t(torch.index_select(tensor, 1, [i]))

    def encode(self, src, src_length, enc_hidden):
        src = src.view(-1, 1)
        # (src_length, 1, inputDim)
        src_embed = self.srcEmbedding(src)
        output, (last_state, last_cell) = self.enc(src_embed, enc_hidden)
        return output, (last_state, last_cell)

    def decoder(self, input, dec_hidden, dec_c0):
        lstm = nn.LSTMCell(input_size=self.inputDim, hidden_size=self.hiddenDim)
        #TODO initialize this LSTMcell
        h1, c1 = lstm(input, (dec_hidden, dec_c0)) # self.dec(input, dec_hidden)
        return h1, c1

    def decoderAction(self, action, h0, c0): # call forward for action LSTM
        h1, c1 = self.act(action, (h0, c0))
        return h1, c1

    def calcContextVec(self):
        contextVec = 0  #TODO create context vector

        return contextVec

    def decoderAttention(self, dec_hidden):
        context_vec = self.calcContextVec()
        torch.cat((dec_hidden, context_vec))
        return F.tanh(self.stildeAffine(dec_hidden)) # s_tilde

    def compositionFunc(self, phraseNum, head, dependent, relation):
        embedVecEnd = torch.cat((head, dependent, relation), 1)
        self.embedVecEnd[phraseNum - self.tgtLen] = embedVecEnd
        embed = self.embedVecAffine(embedVecEnd)

        return F.tanh(embed)

    def decoderReduceLeft(self, tgt, phraseNum, actNum, k, train, actions):
        top = self.reduceHeadStack(k)
        rightNum, leftNum = self.reduceStack()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)

        if rightNum < self.tgtLen and leftNum < self.tgtLen: # word embedding
            # head = self.targetEmbed.col(data.tgt[rightNum])
            # dependent = self.targetEmbed.col(data.tgt[leftNum])
            # relation = self.self.actionEmbed.col(data.action[actNum])
            head = self.col(self.targetEmbed, tgt[rightNum]) # parent: right
            dependent = self.col(self.targetEmbed, tgt[leftNum])# child: left
            # self.compositionFunc(self.embedVec[phraseNum - self.tgtLen],
            #                      self.targetEmbed.col(data.tgt[rightNum]),
            #                      self.targetEmbed.col(data.tgt[leftNum]),
            #                      self.actionEmbed.col(data.action[actNum]),
            #                      self.embedVecEnd[phraseNum - self.tgtLen])
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen

            head = self.embedVec[rightNum]              # parent: right
            dependent = self.col(self.targetEmbed, tgt[leftNum]) # child: left
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen

            head = self.col(self.targetEmbed, tgt[rightNum])        # parent: right
            dependent = self.embedVec[leftNum]                      # child: left
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

        relation = self.col(self.actionEmbed, actions[actNum])
        self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(phraseNum, head, dependent, relation)
        #TODO fix above line

        self.outBuf(self.embedVec[phraseNum - self.tgtLen])
        self.embedStack.append(phraseNum)
        return

    def decoderReduceRight(self, tgt, phraseNum, actNum, k, train, actions):
        top = self.reduceHeadStack(k)
        rightNum, leftNum = self.reduceStack()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)

        if rightNum < self.tgtLen and leftNum < self.tgtLen: # word embedding
            head = self.col(self.targetEmbed, tgt[leftNum])
            dependent = self.col(self.targetEmbed, tgt[rightNum])
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen
            head = self.col(self.targetEmbed, tgt[leftNum]) # parent: left
            dependent = self.embedVec[rightNum] # child: right
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen
            head = self.embedVec[leftNum]  # parent: left
            dependent = self.col(self.targetEmbed, tgt[rightNum])  # child: right
        else:
            rightNum -= self.tgtLen
            leftNum -= self.tgtLen

            head = self.embedVect[leftNum]  # parent: left
            dependent = self.embedVec[rightNum]  # child: right

        relation = self.col(self.actionEmbed, actions[actNum])
        self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(phraseNum, head, dependent, relation)
        # TODO fix above line
        self.outBuf(self.embedVec[phraseNum - self.tgtLen]) #sLSTM push (forward)
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

class StackLSTM(nn.Module):
    class EmptyStackError(Exception):
        def __init__(self):
            super().__init__('stack is already empty')

    BATCH_SIZE = 1
    SEQ_LEN = 1

    # Constructor
    # Inheriting torch.nn.Module
    # make sLSTM using pytorch LSTM
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.,
                 bidirectional: bool=False):
        if input_size <= 0:
            raise ValueError(f'nonpositive input size: {input_size}')
        if hidden_size <= 0:
            raise ValueError(f'nonpositive hidden size: {hidden_size}')
        if num_layers <= 0:
            raise ValueError(f'nonpositive number of layers: {num_layers}')
        if dropout < 0. or dropout >= 1.:
            raise ValueError(f'invalid dropout rate: {dropout}')

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)
        self.h0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        #초기값을 random이 아니라 affine으로 잡아야 할것같다.
        init_states = (self.h0, self.c0)
        self._states_hist = [init_states]
        self._outputs_hist = []  # type: List[Variable]

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if name.startswith('weight'):
                init.orthogonal(param)
            else:
                assert name.startswith('bias')
                init.constant(param, 0.)
        init.constant(self.h0, 0.)
        init.constant(self.c0, 0.)

    def forward(self, inputs):
        if inputs.size() != (self.input_size,):
            raise ValueError(f'expected input to have size ({self.input_size},), got {tuple(inputs.size())}')
        assert self._states_hist

        # Set seq_len and batch_size to 1
        inputs = inputs.view(self.SEQ_LEN, self.BATCH_SIZE, inputs.numel())
        next_outputs, next_states = self.lstm(inputs, self._states_hist[-1])
        self._states_hist.append(next_states)
        self._outputs_hist.append(next_outputs)
        return next_states

    def push(self, *args, **kwargs):
        return self(*args, **kwargs)

    def pop(self):
        if len(self._states_hist) > 1:
            self._outputs_hist.pop()
            return self._states_hist.pop()
        else:
            raise self.EmptyStackError()

    @property
    def top(self):
        # outputs: hidden_size
        return self._outputs_hist[-1].squeeze() if self._outputs_hist else None

    def __repr__(self) -> str:
        res = ('{}(input_size={input_size}, hidden_size={hidden_size}, '
               'num_layers={num_layers}, dropout={dropout})')
        return res.format(self.__class__.__name__, **self.__dict__)

    def __len__(self):
        return len(self._outputs_hist)

class Data:
    def __init__(self, _train_src, _train_tgt, _train_action):
        self.train_src = _train_src
        self.train_tgt = _train_tgt
        self.train_action = _train_action
