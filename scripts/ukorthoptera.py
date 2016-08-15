# -*- coding: utf-8 -*-

from os.path import expanduser as home
from classispecies import settings

settings.modelname = "ukorthoptera"
settings.MULTILABEL = False
settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = True
settings.FORCE_FEATXTRALL = True
settings.MULTICORE = True
settings.n_segments = None
settings.DUMP_REPORT = False
settings.normalise = True


SOUNDPATHS = {'default' : {
                   'train' : home('~/Dropbox/Shared/Orthoptera Sound App/species_recordings'),
                   'test'  : ''
                }
            }
    
from classispecies.utils import makelabels

soundfiles_ = makelabels.list_soundfiles(SOUNDPATHS)
train_labels_csv, test_labels_csv = makelabels.make_labels_from_list_of_files(*soundfiles_)

settings.LABELS = {'default' : {
   'train' : train_labels_csv,
   'test'  : test_labels_csv
}}
                            
## Run one only 
settings.FEATURES_PLOT = False
settings.FEATURE_ONEFILE_PLOT = False
settings.sec_segments = 2.0
 
settings.classifier = 'randomforest' # decisiontree, randomforest
settings.analyser = "multiple"
settings.extract_dct   = False
settings.extract_dolog = False
settings.extract_fft2  = True
settings.extract_mel   = False
settings.FORCE_FEATXTR = True
settings.FORCE_FEATXTRALL = True
settings.agg           = "logmod"

settings.FORCE_MULTIRUNNER_RECOMPUTE = True
settings.downscale_factor = 50
settings.highpass_cutoff  = 0
settings.mod_cutoff       = 50
settings.RANDOM_SEED      = 1234

settings.CONFUSION_PLOT   = True

from classispecies.classispecies import Classispecies
class UkOrthopteraModel(Classispecies):
    pass
 
model = UkOrthopteraModel()
model.run()
# model.save_to_db()

### Multirun

# from classispecies.classispecies import multirunner, multiextracter

#settings.SAVE_TO_DB = True
#model = multirunner(UkOrthopteraModel, [None, ], iters=1)

# settings.SPLIT_TRAINING_SET = False
# model = multiextracter(UkOrthopteraModel, [None, ], iters=1)
