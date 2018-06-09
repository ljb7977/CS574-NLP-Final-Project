import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import utils
from torch.autograd import Variable


class NMT_RNNG(nn.Module):
    def __init__(self,
                 sourceVoc,
                 targetVoc,
                 actionVoc,
                 deprelVoc,
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
                 saveDirName,
                 useGCN,
                 gcnDim):
        super(NMT_RNNG, self).__init__()
        self.sourceVoc = sourceVoc
        self.targetVoc = targetVoc
        self.actionVoc = actionVoc
        self.deprelVoc = deprelVoc
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
        self.useGCN = useGCN
        self.gcnDim = gcnDim

        # TODO init
        self.stack = Stack()
        self.headList = []
        self.embedList = []
        self.actState = []

        self.headStack = Stack()
        self.headList = []
        self.embedStack = Stack()
        self.embedList = []

        # embedding
        self.srcEmbedding = nn.Embedding(len(self.sourceVoc.tokenList), self.inputDim)
        self.actEmbedding = nn.Embedding(len(self.actionVoc.tokenList), self.inputActDim)
        self.tgtEmbedding = nn.Embedding(len(self.targetVoc.tokenList), self.inputDim)

        # encoder
        self.enc = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenEncDim, bidirectional=True)
        utils.lstm_init_uniform_weights(self.enc, self.scale)
        utils.set_forget_bias(self.enc, 1.0)

        ################################################################
        '''
        GCN
        '''
        if useGCN:
            self.num_labels = len(self.deprelVoc.tokenList)

            self.W_in = Variable(torch.FloatTensor(self.inputDim, self.gcnDim), requires_grad=True)
            nn.init.xavier_normal(self.W_in)

            self.b_in_list = []
            for i in range(self.num_labels):
                b_in = Variable(torch.FloatTensor(1, self.gcnDim), requires_grad=True)
                nn.init.constant(b_in, 0)
                self.b_in_list.append(b_in)

            self.W_in_gate = Variable(torch.FloatTensor(1, self.gcnDim), requires_grad=True)
            nn.init.uniform(self.W_in_gate)

            self.b_in_gate_list = []
            for i in range(self.num_labels):
                b_in_gate = Variable(torch.FloatTensor(1, self.gcnDim), requires_grad=True)
                nn.init.constant(b_in_gate, 1)
                self.b_in_gate_list.append(b_in_gate)

            self.W_out = Variable(torch.FloatTensor(self.inputDim, self.gcnDim), requires_grad=True)
            nn.init.xavier_normal(self.W_out)

            self.b_out_list = []
            for i in range(self.num_labels):
                b_out = Variable(torch.FloatTensor(1, self.gcnDim), requires_grad=True)
                nn.init.constant(b_out, 0)
                self.b_out_list.append(b_out)

            self.W_out_gate = Variable(torch.FloatTensor(1, self.gcnDim), requires_grad=True)
            nn.init.uniform(self.W_out_gate)

            self.b_out_gate_list = []
            for i in range(self.num_labels):
                b_out_gate = Variable(torch.FloatTensor(1, self.gcnDim), requires_grad=True)
                nn.init.constant(b_out_gate, 1)
                self.b_out_gate_list.append(b_out_gate)

            self.W_self_loop = Variable(torch.FloatTensor(self.inputDim, self.gcnDim), requires_grad=True)
            nn.init.xavier_normal(self.W_self_loop)

            self.W_self_loop_gate = Variable(torch.FloatTensor(1, self.gcnDim), requires_grad=True)
            nn.init.uniform(self.W_self_loop_gate)
        ################################################################

        # decoder
        self.dec = nn.LSTMCell(input_size=self.hiddenDim, hidden_size=self.hiddenDim)
        # utils.lstm_init_uniform_weights(self.dec, self.scale)
        # utils.set_forget_bias(self.enc, 1.0)

        # action
        self.act = nn.LSTM(input_size=self.inputActDim, hidden_size=self.hiddenActDim)
        utils.lstm_init_uniform_weights(self.act, self.scale)
        utils.set_forget_bias(self.act, 1.0)

        self.outBufCell = nn.LSTMCell(input_size=self.inputDim, hidden_size=self.hiddenDim)

        # out buffer, RNNG stack
        # self.outBuf = StackLSTM(input_size=self.inputDim, hidden_size=self.hiddenDim)
        # utils.lstm_init_uniform_weights(self.outBuf, self.scale)
        # utils.set_forget_bias(self.outBuf, 1.0)

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
        # stilde Affine: after attention
        self.attnScore = nn.Linear(self.hiddenDim, self.hiddenEncDim * 2)  # TODO dimension setting
        utils.linear_init(self.attnScore, self.scale)

        self.embedVec = [None] * 150
        self.embedVecAffine = nn.Linear(self.inputDim * 2 + self.inputActDim, self.inputDim)
        utils.linear_init(self.embedVecAffine, self.scale)

        self.actionPredAffine = nn.Linear(self.hiddenDim, len(self.actionVoc.tokenList))
        utils.linear_init(self.actionPredAffine, self.scale)
        self.wordPredAffine = nn.Linear(self.hiddenDim, len(self.targetVoc.tokenList))
        utils.linear_init(self.wordPredAffine, self.scale)
        # for automatic tuning
        self.prevPerp = float('inf')

    def enc_init_hidden(self):
        weight = next(self.parameters()).data
        return (Variable(weight.new(2, 1, self.hiddenEncDim).zero_()),
                Variable(weight.new(2, 1, self.hiddenEncDim).zero_()))

    def forward(self, src, tgt, actions, deprels, src_length, enc_hidden, train=True):
        deprel_dict = {}
        for (labelIdx, head, num) in deprels:
            deprel_dict[str((head, num))] = labelIdx
        uts = []
        s_tildes = []
        self.actionEmbed = self.actEmbedding(actions)
        self.targetEmbed = self.tgtEmbedding(tgt)
        self.sourceEmbed = self.srcEmbedding(src)
        if self.useGCN:
            wordEmbeds = torch.chunk(self.sourceEmbed, src_length)
            rows = []
            for i, wordembed in enumerate(wordEmbeds):
                # print(wordembed.shape)  # (1, inputDim)
                row = torch.mm(wordembed, self.W_self_loop) * self.W_self_loop_gate
                for j in range(src_length):
                    ijstr = str((i+1, j+1))
                    jistr = str((j+1, i+1))
                    if ijstr in deprel_dict:
                        labelIdx = deprel_dict[ijstr]   # i->j labelIdx, out 관련 계산
                        row += (torch.mm(wordembed, self.W_out) + self.b_out_list[labelIdx]) * self.W_out_gate + self.b_out_gate_list[labelIdx]
                    elif jistr in deprel_dict:
                        labelIdx = deprel_dict[jistr]   # i<-j labelIdx, in 관련 계산
                        row += (torch.mm(wordembed, self.W_in) + self.b_in_list[labelIdx]) * self.W_in_gate + self.b_in_gate_list[labelIdx]
                rows.append(row)
            self.sourceEmbed = torch.cat(rows, 0)

        output, (enc_h1, enc_c1) = self.encode(self.sourceEmbed, enc_hidden)
        enc_output = output.view(-1, self.hiddenEncDim * 2)
        # print(enc_output.shape) #(senLen, 2*hiddenEncDim)
        j, k, top = 0, 0, 0
        self.headStack.push(k)
        k += 1
        self.headStack.push(k)
        phraseNum = len(tgt)
        self.tgtLen = len(tgt)

        dec_h1 = F.relu(self.decInitAffine(enc_h1.view(1, self.hiddenEncDim * 2))).view(1, -1)
        dec_c1 = torch.rand(1, self.hiddenDim)
        context_vec = self.calcContextVec(dec_h1, enc_output)
        s_tilde = self.decoderAttention(dec_h1, context_vec)
        s_tildes.append(s_tilde)

        act_h1 = F.relu(self.actInitAffine(enc_h1.view(1, self.hiddenEncDim * 2))).view(1, 1, -1)
        act_c1 = torch.rand(1, 1, self.hiddenActDim)

        self.outBuf = [(torch.rand(1, self.hiddenDim), torch.rand(1, self.hiddenDim))]  # 혹은 zero
        # outBuf = nn.LSTMCell(input_size=self.inputDim, hidden_size=self.hiddenDim)
        h1, c1 = self.outBufCell(self.targetEmbed[j].view(1, -1), self.outBuf[k - 1])
        self.outBuf.append((h1, c1))  # add h and c to outBuf[k]

        self.embedStack.push(j)

        out_h1 = self.outBuf[k - 1][0]
        utEnd = torch.cat((dec_h1, out_h1, act_h1.view(1, -1)), 1)
        ut = F.tanh(self.utAffine(utEnd))
        uts.append(ut)
        j+=1
        k+=1

        for i in range(1, len(actions)):
            actNum = actions[i]
            actout, (act_h1, act_c1) = self.act(self.actionEmbed[i-1].view(1, 1, -1), (act_h1, act_c1))
            # act_h1, act_c1 = self.decoderAction(self.actionEmbed[i - 1])  # put prev action
            if self.getAction(actNum) == 0:  # shift action
                self.headStack.push(k)  # push to headStack
                dec_h1, dec_c1 = self.decoder(s_tilde, dec_h1, dec_c1)  # TODO decoder forward 1 step with stilde
                context_vec = self.calcContextVec(dec_h1, enc_output)
                s_tilde = self.decoderAttention(dec_h1, context_vec)

                # if(j - 1 < len(self.targetEmbed)):
                #print(j, k, len(self.targetEmbed))
                h1, c1 = self.outBufCell(self.targetEmbed[j].view(1, -1), self.outBuf[k - 1])
                self.outBuf.append((h1, c1))  # add h and c to outBuf[k]
                # else:
                #     self.outBuf.push(torch.zeros(self.hiddenDim))
                self.embedStack.push(j)
                s_tildes.append(s_tilde)
                j += 1
            elif self.getAction(actNum) == 1:  # Reduce left
                self.decoderReduceLeft(phraseNum, i - 1, k, True)
                phraseNum += 1
            elif self.getAction(actNum) == 2:  # reduce right
                self.decoderReduceRight(phraseNum, i - 1, k, True)
                phraseNum += 1
            else:
                # continue
                raise ("Action Error: undefined Action")

            out_h1 = self.outBuf[k - 1][0]
            utEnd = torch.cat((dec_h1, out_h1, act_h1.view(1, -1)), 1)
            ut = F.tanh(self.utAffine(utEnd))
            uts.append(ut)

            k += 1

        uts = self.actionPredAffine(torch.stack(uts))
        s_tildes = self.wordPredAffine(torch.stack(s_tildes))
        return uts, s_tildes

    def getAction(self, actNum):
        return self.actionVoc.tokenList[actNum][2]

    def col(self, tensor, i):
        return torch.t(torch.index_select(tensor, 1, i))

    def encode(self, src_embed, enc_hidden):
        output, (last_state, last_cell) = self.enc(src_embed.view(-1, 1, self.inputDim), enc_hidden)
        return output, (last_state, last_cell)

    def decoder(self, input, h0, c0):
        # lstm = nn.LSTMCell(input_size=self.hiddenDim, hidden_size=self.hiddenDim)
        # #TODO initialize this LSTMcell
        # if args:
        #     h0, c0 = args[:2]
        #     h1, c1 = lstm(input, (h0, c0))  # self.dec(input, dec_hidden)
        # else:
        #     h1, c1 = lstm(input)
        h1, c1 = self.dec(input, (h0, c0))
        return h1, c1

    def decoderAction(self, action):  # call forward for action LSTM
        # if args:
        #     h0, c0 = args[0], args[1]
        #     h1, c1 = self.act(action, (h0, c0))
        # else:
        h1, c1 = self.act(action)
        return h1, c1

    def calcContextVec(self, dec_h1, enc_output):
        temp = self.attnScore(dec_h1)

        attention_score = torch.matmul(temp, torch.t(enc_output))

        alpha = F.softmax(attention_score, dim=1)
        context_vec = torch.matmul(alpha, enc_output)
        return context_vec

    def decoderAttention(self, dec_hidden, context_vec):  # calc context vector and concat, linear and tanh
        # print("h0, c", dec_hidden.shape, context_vec.shape)
        dec_hidden = torch.cat((dec_hidden, context_vec), 1)
        # print("concat", dec_hidden.shape)
        return F.relu(self.stildeAffine(dec_hidden))  # return s_tilde

    def compositionFunc(self, head, dependent, relation):
        embedVecEnd = torch.cat((head, dependent, relation))
        return F.tanh(self.embedVecAffine(embedVecEnd))

    def decoderReduceLeft(self, phraseNum, actNum, k, train):
        top = self.headStack.reduceHead(k)
        rightNum, leftNum = self.embedStack.reduce()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)

        if rightNum < self.tgtLen and leftNum < self.tgtLen:  # word embedding
            head = self.targetEmbed[rightNum]  # parent: right
            dependent = self.targetEmbed[leftNum]  # child: left
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen
            head = self.embedVec[rightNum]  # parent: right
            dependent = self.targetEmbed[leftNum]  # child: left
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen
            head = self.targetEmbed[rightNum]  # parent: right
            dependent = self.embedVec[leftNum]  # child: left
        else:
            rightNum -= self.tgtLen
            leftNum -= self.tgtLen
            head = self.embedVec[rightNum]  # parent: right
            dependent = self.embedVec[leftNum]  # child: left

        relation = self.actionEmbed[actNum]
        self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(head, dependent, relation)

        # outBuf = nn.LSTMCell(input_size=self.inputDim, hidden_size=self.hiddenDim)
        h1, c1 = self.outBufCell(self.embedVec[phraseNum - self.tgtLen].view(1, -1), self.outBuf[top])
        self.outBuf.append((h1, c1))  # add h and c to outBuf[k]
        # print(self.outBuf)
        self.embedStack.push(phraseNum)
        return

    def decoderReduceRight(self, phraseNum, actNum, k, train):
        top = self.headStack.reduceHead(k)
        rightNum, leftNum = self.embedStack.reduce()

        if train:
            self.headList.append(top)
            self.embedList.append(leftNum)
            self.embedList.append(rightNum)

        if rightNum < self.tgtLen and leftNum < self.tgtLen:  # word embedding
            head = self.targetEmbed[leftNum]
            dependent = self.targetEmbed[rightNum]
        elif rightNum > (self.tgtLen - 1) and leftNum < self.tgtLen:
            rightNum -= self.tgtLen
            head = self.targetEmbed[leftNum]  # parent: left
            dependent = self.embedVec[rightNum]  # child: right
        elif rightNum < self.tgtLen and leftNum > (self.tgtLen - 1):
            leftNum -= self.tgtLen
            head = self.embedVec[leftNum]  # parent: left
            dependent = self.targetEmbed[rightNum]  # child: right
        else:
            rightNum -= self.tgtLen
            leftNum -= self.tgtLen
            head = self.embedVec[leftNum]  # parent: left
            dependent = self.embedVec[rightNum]  # child: right

        relation = self.actionEmbed[actNum]
        self.embedVec[phraseNum - self.tgtLen] = self.compositionFunc(head, dependent, relation)

        # outBuf = nn.LSTMCell(input_size=self.inputDim, hidden_size=self.hiddenDim)
        h1, c1 = self.outBufCell(self.embedVec[phraseNum - self.tgtLen].view(1, -1), self.outBuf[top])
        self.outBuf.append((h1, c1))  # add h and c to outBuf[k]
        self.embedStack.push(phraseNum)
        return

    def translate(self, src):
        s_tildes = []
        self.sourceEmbed = self.srcEmbedding(src)
        enc_h0, enc_c0 = torch.zeros(2, 1, self.hiddenEncDim), torch.zeros(2, 1, self.hiddenEncDim)
        output, (enc_h1, enc_c1) = self.encode(self.sourceEmbed, (enc_h0, enc_c0))
        enc_output = output.view(-1, self.hiddenEncDim * 2)

        s_tilde = enc_h1.view(1, self.hiddenEncDim * 2)
        dec_h1 = F.relu(self.decInitAffine(enc_h1)).view(1, -1)
        dec_c1 = torch.rand(self.hiddenDim).view(1, -1)

        for i in range(100):  # maxlen
            dec_h1, dec_c1 = self.decoder(s_tilde, dec_h1, dec_c1)  # decoder forward 1 step with stilde
            context_vec = self.calcContextVec(dec_h1, enc_output)
            s_tilde = self.decoderAttention(dec_h1, context_vec)
            s_tildes.append(s_tilde)

        s_tildes = self.wordPredAffine(torch.stack(s_tildes))
        return s_tildes


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
                 bidirectional: bool = False):
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
        # 초기값을 random이 아니라 affine으로 잡아야 할것같다.
        init_states = (self.h0, self.c0)
        self._states_hist = [init_states]
        # self._outputs_hist = []  # type: List[Variable]

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
        # self._outputs_hist.append(next_outputs)
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
