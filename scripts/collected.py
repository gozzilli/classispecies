'''
Created on 27 Jan 2015

@author: Davide Zilli
'''

from classispecies import settings

settings.modelname  = "collected"
settings.MULTILABEL = False
settings.SPLIT_TRAINING_SET = False
settings.FORCE_FEATXTR = True
settings.FORCE_FEATXTRALL = True
settings.MULTICORE = False
settings.n_segments = None
settings.logger_level = "INFO"

    

settings.LABELS = {'default' : {
   'train' : settings.modelname + "-train.csv",
   'test'  : settings.modelname + "-test.csv"
}}


settings.FEATURES_PLOT = True
settings.FEATURE_ONEFILE_PLOT = True
settings.sec_segments = 1.0

settings.classifier = 'randomforest' #"decisiontree"
settings.analyser   = 'hertzfft'

from classispecies.classispecies import Classispecies, multirunner
class CollectedOrthopteraModel(Classispecies):
    pass

settings.FORCE_FEATXTR = True
settings.FORCE_FEATXTRALL = True

#for _ in range(1):
#    model = CollectedOrthopteraModel()
#    model.run()
#    #model.save_to_db()

multirunner(CollectedOrthopteraModel, [0.5, 1.0, None])