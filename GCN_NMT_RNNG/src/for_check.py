import Translator

t = Translator.Translator(srcTrain_tagged='../data/tagged_train.kr',
                          tgtTrain_tagged='../data/tagged_train.en',
                          srcDev_tagged='../data/tagged_dev.kr',
                          tgtDev_tagged='../data/tagged_dev.en',
                          srcVocaThreshold=1,
                          tgtVocaThreshold=1,
                          deprelLabelThreshold=1)

t.demo(inputDim=512,
       inputActDim=128,
       hiddenDim=256,
       hiddenEncDim=256,
       hiddenActDim=256,
       scale=0.1,
       clipThreshold=10,
       beamSize=10,
       maxLen=80,
       miniBatchSize=128,
# =======
#        hiddenDim=512,
#        hiddenEncDim=512,
#        hiddenActDim=128,
#        scale=0.5,
#        clipThreshold=10,
#        beamSize=10,
#        maxLen=80,
#        miniBatchSize=1025,
# >>>>>>> 5dfd9778946e3f8252fd728068237860e6cceee1
       threadNum=5,
       learningRate=0.001,
       saveDirName='../save/',
       loadModelName='',
       loadGradName='',
       startIter=5,
       epochs=10,
       useGCN=True,
       gcnDim=512)


















