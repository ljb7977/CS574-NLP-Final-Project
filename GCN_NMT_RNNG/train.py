import sys
sys.path.insert(0, './src')

import Translator

#   trainSize < 100000
#   testSize < 10001
#   devSize < 10001
t = Translator.Translator(mode='train',
                          prepocessed=False,
                          srcVocaThreshold=1,
                          tgtVocaThreshold=1,
                          deprelLabelThreshold=1,
                          printEvery=50,
                          trainSize=30000,
                          testSize=500,
                          devSize=300)

t.demo(inputDim=256,
       inputActDim=128,
       hiddenDim=256,
       hiddenEncDim=256,
       hiddenActDim=128,
       scale=0.1,
       miniBatchSize=256,
       learningRate=0.001,
       loadModel=False,
       modelDir='./data/saved_model/',
       modelName='model.pt',
       startIter=1,
       epochs=10,
       useGCN=True,
       gcnDim=256)































