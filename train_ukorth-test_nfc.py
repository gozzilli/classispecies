import os
from classispecies import settings
from classispecies.utils import misc
import numpy as np
from sklearn.preprocessing import label_binarize
from cicada_dataset import cicada_dataset

settings.modelname  = "Train UK Orth - Test NFC dataset"
settings.classifier = "randomforest"
settings.analyser   = "oskmeans" # "mel-filterbank" #"mfcc" #"hertz-spectrum"
settings.MULTILABEL = False
settings.sec_segments = 2.0
settings.n_segments = None
settings.FORCE_FEATXTR = True
settings.SPLIT_TRAINING_SET = False

settings.extract_mean = True
settings.extract_std  = True

settings.SOUNDPATHS.update(
    {'Boa': {
        "train": os.path.expanduser('~/Dropbox/Shared/Orthoptera Sound App/species_recordings'),
        "test" : None
    }})
    
mel_feat = None

from classispecies.classispecies import Classispecies
class MixedInsectsModel(Classispecies):
    ''' train on UK orthoptera, test on cicadas, dark and roesel bush-crickets '''

    
    def load_labels(self, min_row=0, max_row=None, min_col=0, max_col=None):
        
        mapping = {"cic" : "99 ", "dar" : "16 ", "roe" : "15 ", "sil": "00 "}
        
        train_labels = np.array([os.path.basename(x)[0:3] for x in self.train_soundfiles], dtype="S3")
        test_labels = np.array([mapping[os.path.basename(x).split("_")[1][0:3]] for x in self.test_soundfiles], dtype="S3")
        
        classes = np.unique(train_labels)
        #bin_labels = label_binarize(named_labels, classes)
        
        print "train labels:"
        print train_labels
        print "test labels:"
        print test_labels        
        print "classes:"
        print classes
        
        return train_labels, test_labels, classes

    def make_labels_from_filename(self, soundfiles):
            
        named_labels  = np.array([os.path.basename(x).split("_")[1][0:3] for x in soundfiles], dtype="S3")
        classes = np.unique(named_labels)
        labels  = label_binarize(named_labels, classes)
        print classes
        
        return labels, classes, named_labels
    
    def list_soundfiles(self):
        ''' select files in soundpath that contain the names is CLASSES '''
        
        train_path = misc.get_path(settings.SOUNDPATHS, "train")
        
        train_soundfiles = [os.path.join(train_path, x) for x in os.listdir(train_path) if x.endswith(".wav")]
        train_soundfiles += [
            '/home/dz2v07/cloud/Copy/PhD/SoundTrap/Resources/insect_sounds/99 cicada_5223.wav',
            '/home/dz2v07/cloud/Copy/PhD/SoundTrap/Resources/insect_sounds/00 silent_4892.wav'
        ]

        test_soundfiles = cicada_dataset
        return train_soundfiles, test_soundfiles

model = MixedInsectsModel()
model.run()