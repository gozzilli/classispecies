import os
from os.path import expanduser as home
from classispecies import settings

settings.modelname  = "ukorthoptera"
settings.MULTILABEL = False
settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = False
settings.MULTICORE = True

settings.SOUNDPATHS['default'] = {
   'train' : home('~/Dropbox/Shared/Orthoptera Sound App/species_recordings'),
   'test'  : ''
}
    
from classispecies.utils import misc, makelabels


train_labels_csv, test_labels_csv = makelabels.make_labels_from_list_of_files(*makelabels.list_soundfiles())
settings.LABELS = {'default' : {
   'train' : train_labels_csv,
   'test'  : test_labels_csv
}}


for analyser in ["hertzfft"]: # "mel-filterbank", "mfcc",
    for classifier in ["decisiontree"]:
        for sec_segments in [0.5]:
            
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