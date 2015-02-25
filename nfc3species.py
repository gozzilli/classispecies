# -*- coding: utf-8 -*-
import os
from classispecies import settings
from pandas import read_csv


settings.modelname  = "nfc3species18"
settings.classifier = "decisiontree"
settings.analyser   = "hertzfft" #"mel-filterbank" #"mfcc" #"hertz-spectrum"
settings.MULTILABEL = False
settings.MULTICORE  = True
settings.sec_segments = None
settings.n_segments = None
settings.FORCE_FEATXTR = False
settings.SPLIT_TRAINING_SET = True
settings.logger_level = "INFO"
settings.RANDOM_SEED = 12345

settings.MODEL_BINARY = False
settings.SPLIT_TRAINING_SET = False
settings.highpass_cutoff = 7000.
settings.mod_cutoff      = 50.


settings.LABELS = {'default' : {
                        'train' : 'cicada-train.csv',
                        'test'  : None}
                   }

from classispecies.classispecies import Classispecies, multirunner,\
    multiextracter
import random
import numpy as np

settings.DUMP_REPORT = False
settings.FEATURES_PLOT = False
settings.FEATURE_ONEFILE_PLOT = False
settings.sec_segments = None
settings.downscale_factor = 50
    
settings.classifier = 'randomforest' #"decisiontree"  # 
settings.analyser = "multiple"
settings.extract_dct = False
settings.extract_dolog = False
     
# settings.extract_fft2 = True
# settings.extract_mean = False
# settings.extract_std  = False
# settings.agg          = "mod0"
# settings.MOD_TAKE1BIN = True 
    
settings.extract_fft2 = True
settings.extract_mean = False
settings.extract_std  = False
settings.agg          = "mod"
settings.MOD_TAKE1BIN = False
     
# settings.extract_fft2 = False
# settings.extract_mean = True
# settings.extract_std  = False
# settings.agg          = u"μ"  
     
# settings.extract_fft2 = False
# settings.extract_mean = True
# settings.extract_std  = True
# settings.agg          = u"μ+σ" 
     
settings.extract_mel = False
settings.FORCE_FEATXTR    = True
settings.FORCE_FEATXTRALL = True
     
settings.extract_max     = False
     
settings.CONFUSION_PLOT = False
settings.MODEL_BINARY = False

class Nfc3SpeciesModel(Classispecies):
    pass

    def load(self, is_multilabel=None, usecols_train=None, usecols_test=None):
     
        train_label_path = settings.LABELS['default']['train']
         
        ### TRAINING SET ###
        if not train_label_path or not os.path.exists(train_label_path):
            raise Exception("Cannot load train labels")
 
         
        labels = read_csv(train_label_path, header=None, dtype="S")
         
        train_labels = labels.ix[:,1].as_matrix()
         
        classes = np.unique(train_labels).astype("S2")
         
        n = 9
        rows_train = []
        rows_test  = []
        for insect in classes:
            random.seed(settings.RANDOM_SEED)
            rows = random.sample(labels[labels[1] == insect].index, n*2)
             
            rows_train += rows[0:n]
            rows_test  += rows[n:]
 
        train_soundfiles = labels[0].ix[rows_train]
        test_soundfiles  = labels[0].ix[rows_test]
     
        train_labels     = np.ravel(labels.ix[rows_train,1].as_matrix())
        test_labels      = np.ravel(labels.ix[rows_test ,1].as_matrix())
         
        return train_soundfiles, train_labels, test_soundfiles, test_labels, classes

class Nfc3SpeciesModelAll(Classispecies):
    pass
     
f1s = []
rocs   = []  
feat_imp = []
truepred = []
iters = 1
 
#     
# model = Nfc3SpeciesModel()
# model.featxtr()

# settings.SAVE_TO_DB = False
#       
# for i in range(iters):
#     print i
#     model = Nfc3SpeciesModel()
#     model.run()
#     f1s.append(model.results["f1"])
#     rocs.append(model.results["auc"])
#     feat_imp.append(model.clf.feature_importances_)
#     truepred.append(model.truepred)
#             
#     settings.FORCE_FEATXTR    = True
#     settings.FORCE_FEATXTRALL = True
#       
# np.savetxt('/tmp/featimp.txt', np.vstack(feat_imp))
# f1s = np.array(f1s)
# print "f1s %s\nmeans: %.3f, std: %.3f, sem: %.3f" % (f1s, np.mean(f1s), np.std(f1s), np.sqrt(np.var(f1s)/iters))
#       
# from classispecies.utils.confusion import ConfusionMatrix
#       
# confusion = ConfusionMatrix(model.classes)
#       
# for t in truepred:
#     for true, pred in t:
#         confusion.add(true, pred)
#               
#       
# print confusion
# confusion.plot('/tmp/allconfusion.png')
# with open("/tmp/conf.txt", 'a') as f:
#     f.write(str(confusion))
#     f.write("\n\n")
#     f.write(confusion.toval())
     
         
# import pickle
# i = 2
# with open ('/tmp/model%i.pickle' % i, 'w') as f:
#     pickle.dump(model, f)
      
 
settings.CONFUSION_PLOT = False
settings.FORCE_MULTIRUNNER_RECOMPUTE = False
 
settings.SAVE_TO_DB = True      
model = multirunner(Nfc3SpeciesModel, [None], iters=1)

# settings.SAVE_TO_DB = False
# settings.FORCE_MULTIRUNNER_RECOMPUTE = True
# settings.SPLIT_TRAINING_SET = True
# model = multirunner(Nfc3SpeciesModelAll, [None], iters=1)

# settings.SPLIT_TRAINING_SET = False
# model = multiextracter(Nfc3SpeciesModel, [None, 1.0, 2.0], iters=1)
