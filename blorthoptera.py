# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:54:41 2015

@author: Davide Zilli
"""

from classispecies import settings

settings.modelname  = "blorthoptera"
settings.MULTILABEL = False
settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = False
settings.MULTICORE = True
settings.FORCE_FEATXTRALL = False

settings.LABELS = {'default' : {
   'train' : settings.modelname+"-train.csv",
   'test'  : None
}}


from classispecies.classispecies import Classispecies, multirunner
class BlOrthopteraModel(Classispecies):
    pass

settings.sec_segments = 5.0
settings.classifier = "decisiontree" #randomforest"
settings.analyser   = "hertzfft" #"melfft" #hertz-spectrum" #"oskmeans" 
            
model = BlOrthopteraModel()
model.run()
#model.dump_to_db()

#multirunner(BlOrthopteraModel)