from operator import itemgetter


class Vocabulary(object):
    def __init__(self, trainFile, tokenFreqThreshold, Action_or_Language):
        if Action_or_Language == 'lang':
            self.initLang(trainFile, tokenFreqThreshold)
        else:
            self.initAction(trainFile)

    def initLang(self, trainFile, tokenFreqThreshold):
        self.tokenList = []     # list of (token, count)
        self.tokenIndex = {}    # token to index
        unkCount = 0
        eosCount = 0
        tokenCount = {}
        tokens = []
        with open(trainFile) as f:
            for line in f:
                eosCount += 1
                tokens += line.split(' \t')
                for token in tokens:
                    if token in tokenCount:
                        tokenCount[token] += 1
                    else:
                        tokenCount[token] = 1
        for token in tokens:
            if tokenCount[token] >= tokenFreqThreshold:
                self.tokenList.append((token, tokenCount[token]))
            else:
                unkCount += tokenCount[token]
        self.tokenList.sort(key=itemgetter(1))
        self.tokenList.reverse()    # (token, count) tuple이 count가 큰 순서로 정렬
        tokenList_len = len(self.tokenList)
        for i in range(tokenList_len):
            self.tokenIndex[self.tokenList[i][0]] = i
        self.eosIndex = tokenList_len
        self.tokenList.append(("*EOS*", eosCount))
        self.unkIndex = self.eosIndex + 1
        self.tokenList.append(("*UNK*", unkCount))

    def initAction(self, trainFile):
        self.tokenList = []  # list of (token, count)
        self.tokenIndex = {}  # token to index
        tokens = []
        tokenCount = {}
        eosCount = 0
        with open(trainFile) as f:
            for line in f:
                eosCount += 1
                tokens += line.split(' \t')
                for token in tokens:
                    if token in tokenCount:
                        tokenCount[token] += 1
                    else:
                        tokenCount[token] = 1
        for token in tokens:
            if "SHIFT" in token:
                self.tokenList.append((token, tokenCount[token], 0))
            elif "LEFT" in token:
                self.tokenList.append((token, tokenCount[token], 1))
            elif "RIGHT" in token:
                self.tokenList.append((token, tokenCount[token], 2))
            else:
                print("Error: Non shift/reduce word.")
        self.tokenList.sort(key=itemgetter(1))
        self.tokenList.reverse()  # (token, count, actionIndex) tuple이 count가 큰 순서로 정렬
        tokenList_len = len(self.tokenList)
        for i in range(tokenList_len):
            self.tokenIndex[self.tokenList[i][0]] = i













































