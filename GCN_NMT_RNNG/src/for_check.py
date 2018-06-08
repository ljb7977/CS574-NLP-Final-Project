import Translator

t = Translator.Translator(srcTrain='../data/new/train.en',
                          tgtTrain='../data/new/train.kr',
                          actTrain='../data/new/train.oracle.kr',
                          srcDev='../data/new/dev.en',
                          tgtDev='../data/new/dev.kr',
                          actDev='../data/new/dev.oracle.kr',
                          srcVocaThreshold=1,
                          tgtVocaThreshold=1)
t.demo(inputDim=128,
       inputActDim=128,
       hiddenDim=128,
       hiddenEncDim=128,
       hiddenActDim=128,
       scale=0.02,
       clipThreshold=10,
       beamSize=10,
       maxLen=10,
       miniBatchSize=64,
       threadNum=5,
       learningRate=0.01,
       saveDirName='../save/',
       loadModelName='',
       loadGradName='',
       startIter=5,
       epochs=10)

'''srcTrain='../data/experiment.kr',
                          tgtTrain='../data/experiment.en',
                          actTrain='../data/experiment.oracle.en','''
