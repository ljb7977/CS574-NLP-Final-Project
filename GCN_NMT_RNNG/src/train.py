import sys
sys.path.insert(0, './src')

import Translator


t = Translator.Translator(mode='train',
                          prepocessed=False,
                          srcVocaThreshold=1,
                          tgtVocaThreshold=1,
                          deprelLabelThreshold=1,
                          printEvery=50)

t.demo(inputDim=256,
       inputActDim=128,
       hiddenDim=256,
       hiddenEncDim=256,
       hiddenActDim=128,
       scale=2,
       miniBatchSize=256,
       learningRate=0.08,
       loadModel=False,
       modelDir='./data/saved_model/',
       modelName='fuck.pt',
       startIter=1,
       epochs=10,
       useGCN=True,
       gcnDim=256)































