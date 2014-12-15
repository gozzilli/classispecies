import os
from classispecies import settings
from classispecies.utils import misc

settings.modelname  = "ukorthoptera"
settings.MULTILABEL = False
settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = True
settings.SOUNDPATHS.update({'default': os.path.expanduser('~/Dropbox/Shared/Orthoptera Sound App/species_recordings'),})

for analyser in ["mel-filterbank", "mfcc"]:
    for classifier in ["decisiontree"]:
        for sec_segments in [0.5, 0.2]:
            
            print "=" * 80
            print "ANALYSER:   %s" % analyser
            print "CLASSIFIER: %s" % classifier
            print "SEC SEGM:   %ss" % sec_segments
            
            settings.classifier = classifier
            settings.analyser   = analyser
            #settings.n_segments = 5
            settings.sec_segments = sec_segments
            settings.NMFCCS = 13
            
            
            from classispecies.classispecies import Classispecies
            class UkOrthopteraModel(Classispecies):
                pass
            
            model = UkOrthopteraModel()
            model.run()