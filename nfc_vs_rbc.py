# -*- coding: utf-8 -*-

from classispecies import settings


settings.modelname  = "nfc_vs_rbc"
settings.classifier = "decisiontree"
settings.analyser   = "hertzfft" #"mel-filterbank" #"mfcc" #"hertz-spectrum"
settings.MULTILABEL = False
settings.MULTICORE  = True
settings.sec_segments = None
settings.n_segments = None
settings.FORCE_FEATXTR = False
settings.SPLIT_TRAINING_SET = True
settings.logger_level = "INFO"

settings.LABELS = {'default' : {
                        'train' : 'nfc_rbc-train.csv',
                        'test'  : None}
                   }

from classispecies.classispecies import Classispecies, multirunner
class NfcRbcModel(Classispecies):
    pass

settings.DUMP_REPORT = False
settings.FEATURES_PLOT = False
settings.FEATURE_ONEFILE_PLOT = False
settings.sec_segments = None
   
settings.classifier = 'randomforest' #"decisiontree"  # 
settings.analyser = "multiple"
settings.extract_dct = False
settings.extract_dolog = False

#settings.extract_fft2 = True
#settings.extract_mean = False
#settings.extract_std  = False
#settings.agg          = "mod"
#settings.MOD_TAKE1BIN = True

#settings.extract_fft2 = False
#settings.extract_mean = True
#settings.extract_std  = False
#settings.agg          = u"μ" # u"μ+σ" 

settings.extract_fft2 = False
settings.extract_mean = True
settings.extract_std  = True
settings.agg          = u"μ+σ" 

settings.extract_mel = False
settings.FORCE_FEATXTR    = True
settings.FORCE_FEATXTRALL = True

settings.extract_max     = False
settings.RANDOM_SEED   = None
settings.downscale_factor = None
settings.CONFUSION_PLOT = False

settings.MODEL_BINARY = True

f1s = []
rocs   = []  
feat_imp = []
truepred = []
iters = 100
for i in range(iters):
    print i
    model = NfcRbcModel()
    model.run()
    f1s.append(model.results["f1"])
    rocs.append(model.results["auc"])
    feat_imp.append(model.clf.feature_importances_)
    truepred.append(model.truepred)
    
    settings.FORCE_FEATXTR    = False
    settings.FORCE_FEATXTRALL = False


import numpy as np
np.savetxt('/tmp/featimp.txt', np.vstack(feat_imp))
f1s = np.array(f1s)
print "f1s %s\nmeans: %s, std: %s, sem: %s" % (None, np.mean(f1s, axis=0), np.std(f1s, axis=0), np.sqrt(np.var(f1s, axis=0)/iters))
#print "f1s %s\nmeans: %.3f, std: %.3f, sem: %.3f" % (f1s, np.mean(f1s), np.std(f1s), np.sqrt(np.var(f1s)/iters))
#print "rocs %s\nmeans: %.3f, std: %.3f, sem: %.3f" % (rocs, np.mean(rocs), np.std(rocs), np.sqrt(np.var(rocs)/iters))
    
#settings.SPLIT_METHOD = "split-before"
#model = multirunner(NfcRbcModel, [None, 0.5, 1.0, 5.0], iters=100)

from classispecies.utils.confusion import ConfusionMatrix

confusion = ConfusionMatrix(model.classes)

for t in truepred:
    for true, pred in t:
        confusion.add(true, pred)
        
print confusion
confusion.plot('/tmp/allconfusion.png')
