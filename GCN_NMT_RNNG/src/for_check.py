import Translator

t = Translator.Translator(srcTrain_tagged='../data/tagged_train.kr',
                          tgtTrain_tagged='../data/tagged_train.en',
                          srcDev_tagged='../data/tagged_dev.kr',
                          tgtDev_tagged='../data/tagged_dev.en',
                          srcVocaThreshold=1,
                          tgtVocaThreshold=1,
                          deprelLabelThreshold=1)
t.demo(inputDim=256,
       inputActDim=128,
       hiddenDim=256,
       hiddenEncDim=256,
       hiddenActDim=256,
       scale=0.1,
       clipThreshold=10,
       beamSize=10,
       maxLen=80,
       miniBatchSize=128,
       threadNum=5,
       learningRate=0.001,
       saveDirName='../save/',
       loadModelName='',
       loadGradName='',
       startIter=5,
       epochs=10,
       useGCN=True)


















