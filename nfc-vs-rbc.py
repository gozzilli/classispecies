import os
from sklearn.preprocessing import label_binarize
import numpy as np

from classispecies import settings
from cicada_dataset import cicada_dataset

settings.modelname  = "nfc_vs_rbc"
settings.classifier = "decisiontree"
settings.analyser   = "mel-filterbank" #"mfcc" #"hertz-spectrum"
settings.MULTILABEL = False
settings.sec_segments = None
settings.n_segments = None
settings.FORCE_FEATXTR = False

settings.SOUNDPATHS.update({'default': os.path.join(settings._here,'../../Resources/cicadasounds'),})

from classispecies.classispecies import Classispecies
class NfcRbcModel(Classispecies):

    def make_labels_from_filename(self, soundfiles):
            
        named_labels  = np.array([os.path.basename(x).split("_")[1][0:3] for x in soundfiles], dtype="S3")
        classes = np.unique(named_labels)
        labels  = label_binarize(named_labels, classes)
        print classes
        
        return labels, classes, named_labels
    
    def list_soundfiles(self):
        ''' select files in soundpath that contain the names is CLASSES '''
        
        return cicada_dataset, None

model = NfcRbcModel()
model.run()