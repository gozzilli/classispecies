import os
from classispecies import settings
from classispecies.utils import misc

settings.modelname  = "ukorthoptera"
settings.classifier = "randomforest"
settings.analyser   = "mfcc" 
settings.MULTILABEL = False
#settings.n_segments = 5
settings.sec_segments = 2.
settings.NMFCCS = 13
settings.FORCE_FEATXTR = False

settings.SOUNDPATHS.update({'default': os.path.expanduser('~/Dropbox/Shared/Orthoptera Sound App/species_recordings'),})

from classispecies.classispecies import Classispecies
class UkOrthopteraModel(Classispecies):
    pass

model = UkOrthopteraModel()
model.run()