import Translator

t = Translator.Translator(srcTrain='../data/train.en',
                          tgtTrain='../data/train.ja',
                          actTrain='../data/train.oracle.en',
                          srcDev='../data/dev.en',
                          tgtDev='../data/dev.ja',
                          actDev='../data/dev.oracle.en',
                          srcVocaThreshold=1,
                          tgtVocaThreshold=1)
t.demo2(inputDim=10,
        inputActDim=10,
        hiddenDim=10,
        hiddenEncDim=10,
        hiddenActDim=10,
        scale=10,
        clipThreshold=10,
        beamSize=10,
        maxLen=10,
        miniBatchSize=10,
        threadNum=5,
        learningRate=0.005,
        saveDirName='../save/',
        loadModelName='',
        loadGradName='',
        startIter=5)






































