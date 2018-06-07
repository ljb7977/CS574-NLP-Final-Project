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
        self.stack = Stack()
        self.headList = []
        self.embedList = []
        self.actState = []

        self.headStack = Stack()
        self.headList = []
        self.embedStack = Stack()
        self.embedList = []

        #embedding
        self.srcEmbedding = nn.Embedding(len(self.sourceVoc.tokenList), self.inputDim)
        self.actEmbedding = nn.Embedding(len(self.actionVoc.tokenList), self.inputActDim)
        self.tgtEmbedding = nn.Embedding(len(self.targetVoc.tokenList), self.hiddenDim)

        # encoder
        self.enc = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenEncDim, bidirectional=True)
        utils.lstm_init_uniform_weights(self.enc, self.scale)
        utils.set_forget_bias(self.enc, 1.0)

        # decoder
        self.dec = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenDim, bidirectional=True)
        utils.lstm_init_uniform_weights(self.dec, self.scale)
        utils.set_forget_bias(self.enc, 1.0)

        # action
        self.act = StackLSTM(input_size=self.inputActDim, hidden_size=self.hiddenActDim)
        #utils.lstm_init_uniform_weights(self.act, self.scale)
        #utils.set_forget_bias(self.act, 1.0)
        # out buffer, RNNG stack
        self.outBuf = StackLSTM(input_size=self.inputDim, hidden_size=self.hiddenDim)
        #utils.lstm_init_uniform_weights(self.outBuf, self.scale)
        #utils.set_forget_bias(self.outBuf, 1.0)

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
        self.stildeAffine = nn.Linear(self.hiddenDim + self.hiddenEncDim * 2, self.hiddenDim)
        utils.linear_init(self.stildeAffine, self.scale)
        #stilde Affine: after attention
        self.attnScore = nn.Linear(self.hiddenDim, self.hiddenEncDim * 2) # TODO dimension setting
        utils.linear_init(self.attnScore, self.scale)

        self.embedVec = [None] * 150
        self.embedVecAffine = nn.Linear(self.inputDim * 2 + self.inputActDim, self.inputDim)
        utils.linear_init(self.embedVecAffine, self.scale)

        self.actionPredAffine = nn.Linear(self.hiddenDim, len(self.actionVoc.tokenList))
        utils.linear_init(self.actionPredAffine, self.scale)
        self.wordPredAffine = nn.Linear(self.hiddenDim, len(self.targetVoc.tokenList))
        utils.linear_init(self.wordPredAffine, self.scale)

        # # embedding matrices는 보통 inputDim*len(Voc)형태로 만들어져야하는데, 일단 보류
        # self.targetEmbed = torch.Tensor(self.inputDim, len(self.targetVoc.tokenList))
        # init.uniform_(self.targetEmbed, 0., self.scale)
        # self.actionEmbed = torch.Tensor(self.inputActDim, len(self.actionVoc.tokenList))
        # init.uniform_(self.actionEmbed, 0., self.scale)

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
        s_tildes = []
        output, (enc_last_state, last_cell) = self.encode(src, src_length, enc_hidden)
        enc_output = output.view(-1, self.hiddenEncDim*2)
        # print(enc_output.shape)
        self.actionEmbed = self.actEmbedding(actions)
        self.targetEmbed = self.tgtEmbedding(tgt)

        # 이 디코더의 output을 action sLSTM과 decoder LSTM에 넣어야 한다.
        j, k, top = 0, 0, 0
        self.headStack.push(k)
        k += 1
        length = len(src)
        phraseNum = len(tgt)
        self.tgtLen = len(tgt)

        # i == 0
        act_h1, act_c1 = self.decoderAction(self.actionEmbed[0])
        # 일단 0일때는 action 0으로 설정
        #처음 액션은 무조건 shift이므로, 그것만 처리한다.

        self.headStack.push(k)
        enc_last_state = enc_last_state.view(1, self.hiddenEncDim * 2)
        dec_h1 = F.tanh(self.decInitAffine(enc_last_state)).view(1, -1)
        #enc_output = F.tanh(self.decInitAffine(enc_output))
        #dec_h1 = enc_output[-1].view(1, -1)
        # dec_h1 = F.tanh(self.decInitAffine(enc_output)).view(1, -1)
        dec_c1 = torch.zeros(self.hiddenDim).view(1, -1)
        context_vec = self.calcContextVec(dec_h1, enc_output)
        s_tilde = self.decoderAttention(dec_h1, context_vec)
        out_h1, out_c1 = self.outBuf(self.targetEmbed[j]) #0번째 target word embedding 넣음a
        self.embedStack.push(j)

        utEnd = torch.cat((dec_h1, out_h1.view(1, -1), act_h1.view(1, -1)), 1)
        utEnds.append(utEnd)
        ut = F.tanh(self.utAffine(utEnd))
        uts.append(ut)

        j+=1
        k+=1

        for i in range(1, len(actions)):
            actNum = actions[i]
            act_h1, act_c1 = self.decoderAction(self.actionEmbed[i-1]) #put prev action

            if self.getAction(actNum) == 0: #shift action
                print("SHIFT")
                self.headStack.push(k) #push to headStack
                dec_h1, dec_c1 = self.decoder(s_tilde, dec_h1, dec_c1) #TODO decoder forward 1 step with stilde
                context_vec = self.calcContextVec(dec_h1, enc_output)
                s_tilde = self.decoderAttention(dec_h1, context_vec)

                if(j - 1 < len(self.targetEmbed)):
                    self.outBuf.push(self.targetEmbed[j - 1]) #outbut forward (push)
                else:
                    self.outBuf.push(torch.zeros(self.hiddenDim))
                self.embedStack.push(j)
                s_tildes.append(s_tilde)
                j+=1
            elif self.getAction(actNum) == 1: # Reduce left
                print("REDUCE_LEFT")
                self.decoderReduceLeft(phraseNum, i-1, k, True)
                phraseNum+=1
            elif self.getAction(actNum) == 2: #reduce right
                print("REDUCE_RIGHT")
                self.decoderReduceRight(phraseNum, i-1, k, True)
                phraseNum+=1
            else:
                raise("Action Error: undefined Action")

            utEnd = torch.cat((dec_h1, out_h1.view(1, -1), act_h1.view(1, -1)), 1)
            utEnds.append(utEnd)
            ut = F.tanh(self.utAffine(utEnd))
            uts.append(ut)

            k+=1

        # # calc softmax of uts adn stildes
        # predicted_actions = []
        # for ut in uts:
        #     print(F.log_softmax(ut, dim=1))
        #     predicted_actions.append(F.softmax(ut, dim=1))
        # #print("action pred", predicted_actions)
        uts = self.actionPredAffine(torch.stack(uts))
        s_tildes = self.wordPredAffine(torch.stack(s_tildes))
        return uts, s_tildes
        #
        # act_loss = 0
        # for ut, action in zip(uts, self.actionEmbed):
        #     act_loss += criterion(ut, action)
        # loss = 0
        # for word, target in zip(s_tildes, self.targetEmbed):
        #     loss += criterion(word, target)
        # return act_loss, loss

    def getAction(self, actNum):
        return self.actionVoc.tokenList[actNum][2]

    def col(self, tensor, i):
        return torch.t(torch.index_select(tensor, 1, i))

    def encode(self, src, src_length, enc_hidden):
        src = src.view(-1, 1)
        # (src_length, 1, inputDim)
        src_embed = self.srcEmbedding(src)
        output, (last_state, last_cell) = self.enc(src_embed, enc_hidden)
        return output, (last_state, last_cell)

    def decoder(self, input, h0, c0):
        lstm = nn.LSTMCell(input_size=self.hiddenDim, hidden_size=self.hiddenDim)
        # #TODO initialize this LSTMcell
        # if args:
        #     h0, c0 = args[:2]
        #     h1, c1 = lstm(input, (h0, c0))  # self.dec(input, dec_hidden)
        # else:
        #     h1, c1 = lstm(input)
        h1, c1 = lstm(input, (h0, c0))
        return h1, c1

    def decoderAction(self, action): # call forward for action LSTM
        # if args:
        #     h0, c0 = args[0], args[1]
        #     h1, c1 = self.act(action, (h0, c0))
        # else:
        h1, c1 = self.act(action)
        return h1, c1

    def calcContextVec(self, dec_h1, enc_output):
        temp = torch.t(self.attnScore(dec_h1))

        attention_score = torch.matmul(enc_output, temp)

        alpha = torch.t(F.softmax(attention_score, dim=1))
        # print(alpha.shape)
        context_vec = torch.matmul(alpha, enc_output)
        return context_vec

    def decoderAttention(self, dec_hidden, context_vec): #calc context vector and concat, linear and tanh
        # print("h0, c", dec_hidden.shape, context_vec.shape)
        dec_hidden = torch.cat((dec_hidden, context_vec), 1)
        # print("concat", dec_hidden.shape)
        return F.tanh(self.stildeAffine(dec_hidden)) # return s_tilde

    def compositionFunc(self, phraseNum, head, dependent, relation):
        embedVecEnd = torch.cat((head, dependent, relation))
        return F.tanh(self.embedVecAffine(embedVecEnd))

    def decoderReduceLeft(self, phraseNum, actNum, k, train):
        top = self.headStack.reduceHead(k)
        rightNum, leftNum = self.embedStack.reduce()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)

        if rightNum < self.tgtLen and leftNum < self.tgtLen: # word embedding
            head = self.targetEmbed[rightNum]                    # parent: right
            # head = self.col(self.targetEmbed, tgt[rightNum])
            dependent = self.targetEmbed[leftNum]                #child: left
            # dependent = self.col(self.targetEmbed, tgt[leftNum])# child: left
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen

            head = self.embedVec[rightNum]              # parent: right
            dependent = self.targetEmbed[leftNum]            # child: left
            # dependent = self.col(self.targetEmbed, tgt[leftNum]) # child: left
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen
            head = self.targetEmbed[rightNum]                # parent: right
            # head = self.col(self.targetEmbed, tgt[rightNum])
            dependent = self.embedVec[leftNum]          # child: left
        else:
            rightNum -= self.tgtLen
            leftNum -= self.tgtLen
            head = self.embedVec[rightNum]         # parent: right
            dependent = self.embedVec[leftNum]      # child: left

        relation = self.actionEmbed[actNum]
        self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(phraseNum, head, dependent, relation)

        self.outBuf.pop()
        self.outBuf.pop()
        self.outBuf.push(self.embedVec[phraseNum - self.tgtLen])
        self.embedStack.push(phraseNum)
        return

    def decoderReduceRight(self, phraseNum, actNum, k, train):
        top = self.headStack.reduceHead(k)
        rightNum, leftNum = self.embedStack.reduce()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)

        if rightNum < self.tgtLen and leftNum < self.tgtLen: # word embedding
            head = self.targetEmbed[leftNum]
            dependent = self.targetEmbed[rightNum]
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen
            head = self.targetEmbed[leftNum]  # parent: left
            dependent = self.embedVec[rightNum] # child: right
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen
            head = self.embedVec[leftNum]  # parent: left
            dependent = self.targetEmbed[rightNum] # child: right
        else:
            rightNum -= self.tgtLen
            leftNum -= self.tgtLen

            head = self.embedVec[leftNum]  # parent: left
            dependent = self.embedVec[rightNum]  # child: right

        relation = self.actionEmbed[actNum]
        self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(phraseNum, head, dependent, relation)
        self.outBuf.pop()
        self.outBuf.pop()
        self.outBuf.push(self.embedVec[phraseNum - self.tgtLen])
        self.embedStack.push(phraseNum)
        return

    # def calcLoss(self, data, train):
    #     loss = 0.0
    #     lossAct = 0.0
    #     j=0, k=0
    #     phraseNum = data.tgt.size()
    #     actNum = None
    #
    #     self.del_embedVec = {}
    #     self.headStack = []
    #     self.headList = {}
    #     self.embedStack = []
    #     self.embedList = {}
    #     # this->biEncode(data, arg, false)
    #
    #     # k == 0
    #     # self.outBufInitAffine.forward(arg.encStateEnd, arg.outBufState[k]->h);
    #     arg.outBufState[k].c = torch.zeros(self.hiddenDim)
    #
    #     arg.headStack.append(k)
    #     k+=1
    #
    #     for i in range(arg.actLen): #SoftmaxAct calculation
    #         actNum = data.action[i]
    #         self.decoderAction(arg, arg.actState, data.action[i - 1], i, False) # PUSH
    #         if self.actionVoc.tokenList[actNum].action == 0: # 0: Shift
    #             arg.headStack.append(k)
    #             #1) Let a decoder proceed one step; PUSH
    #             self.decoder(arg, arg.decState, arg.s_tilde[j - 1], data.tgt[j - 1], j, False)
    #
    #             # Attention
    #             self.decoderAttention(arg, arg.decState[j], arg.contextSeqList[j], arg.s_tilde[j], arg.stildeEnd[j])
    #             if not self.useBlackOut:
    #                 self.softmax.calcDist(arg.s_tilde[j], arg.targetDist)
    #                 loss += self.softmax.calcLoss(arg.targetDist, data.tgt[j])
    #             else:
    #                 if train: #word prediction
    #                     arg.blackOutState[0].sample[0] = data.tgt[j];
    #                     arg.blackOutState[0].weight.col(0) = self.blackOut.weight.col(data.tgt[j])
    #                     arg.blackOutState[0].bias.coeffRef(0, 0) = self.blackOut.bias.coeff(data.tgt[j], 0)
    #
    #                     self.blackOut.calcSampledDist2(arg.s_tilde[j], arg.targetDist, arg.blackOutState[0])
    #                     loss += self.blackOut.calcSampledLoss(arg.targetDist) #Softmax
    #                 else: # Test Time
    #                     self.blackOut.calcDist(arg.s_tilde[j], arg.targetDist) #Softmax
    #                     loss += self.blackOut.calcLoss(arg.targetDist, data.tgt[j]) #Softmax
    #             # 2) Let the output buffer proceed one step, though the computed unit is not used at this step; PUSH
    #             self.outBuf.forward(self.targetEmbed.col(data.tgt[j]),
    #                                 arg.outBufState[k - 1], arg.outBufState[k])
    #
    #             arg.embedStack.append(j)
    #
    #             # SoftmaxAct calculation(o: output buffer, s: stack, and h: action)
    #             arg.utEnd[0].segment(0, self.hiddenDim) = arg.decState[j].h
    #             j+=1
    #         elif self.actionVoc.tokenList[actNum].action == 1: # 1: Reduce - Left
    #             self.decoderReduceLeft(data, arg, phraseNum, i - 1, k, False)
    #             phraseNum+=1
    #
    #             # SoftmaxAct calculation(o: output buffer, s: stack, and h: action)
    #             arg.utEnd[0].segment(0, self.hiddenDim) = arg.decState[j - 1].h;
    #
    #         elif self.actionVoc.tokenList[actNum].action == 2: # 2: Reduce - Right
    #             self.decoderReduceRight(data, arg, phraseNum, i - 1, k, False)
    #             phraseNum+=1
    #
    #             # SoftmaxAct calculation(o: output buffer, s: stack, and h: action)
    #             arg.utEnd[0].segment(0, this->hiddenDim) = arg.decState[j - 1].h
    #
    #         else:
    #             raise("Error Non-Shift/Reduce")
    #
    #         arg.utEnd[0].segment(self.hiddenDim, self.hiddenDim) = arg.outBufState[k - 1].h
    #         arg.utEnd[0].segment(self.hiddenDim * 2, self.hiddenActDim) = arg.actState[i].h
    #         self.utAffine.forward(arg.utEnd[0], arg.ut[0])
    #
    #         self.softmaxAct.calcDist(arg.ut[0], arg.actionDist)
    #         lossAct += self.softmaxAct.calcLoss(arg.actionDist, data.action[i])
    #
    #         k+=1
    #
    #     arg.clear()
    #     return [loss, lossAct]

class Stack():
    def __init__(self):
        self.stack = []

    def reduce(self):
        right = self.stack.pop()
        left = self.stack.pop()
        return right, left

    def reduceHead(self, k):
        self.stack.pop()
        self.stack.pop()
        top = self.stack[-1]
        self.stack.append(k)
        return top

    def push(self, item):
        self.stack.append(item)

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