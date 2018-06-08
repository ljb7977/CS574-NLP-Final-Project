import Translator

t = Translator.Translator(srcTrain='../data/train.kr',
                          tgtTrain='../data/train.en',
                          actTrain='../data/train.oracle.en',
                          srcDev='../data/dev.kr',
                          tgtDev='../data/dev.en',
                          actDev='../data/dev.oracle.en',
                          srcVocaThreshold=1,
                          tgtVocaThreshold=1)

# t = Translator.Translator(srcTrain='../data/train.ja',
#                           tgtTrain='../data/train.en',
#                           actTrain='../data/train.oracle.en',
#                           srcDev='../data/dev.ja',
#                           tgtDev='../data/dev.en',
#                           actDev='../data/dev.oracle.en',
#                           srcVocaThreshold=1,
#                           tgtVocaThreshold=1)
t.demo(inputDim=256,
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
       learningRate=0.0001,
       saveDirName='../save/',
       loadModelName='',
       loadGradName='',
       startIter=5,
       epochs=10)

'''srcTrain='../data/experiment.kr',
                          tgtTrain='../data/experiment.en',
                          actTrain='../data/experiment.oracle.en','''
