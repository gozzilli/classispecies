# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:54:41 2015

@author: dz2v07
"""

from os.path import expanduser as home
from classispecies import settings
import pandas as pd
import re
import os

settings.modelname  = "blorthoptera"
settings.MULTILABEL = False
settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = False
settings.MULTICORE = True
settings.FORCE_FEATXTRALL = False

settings.SOUNDPATHS['default'] = {
   'train' : home('~/cicada-largefiles/bl-files/all-16bit'),
   'test'  : ''
}


dir_ = settings.SOUNDPATHS['default']['train']
available_files = os.listdir(dir_)

FILES2LOWER = False   

if FILES2LOWER:
    
    for filename in available_files:
        os.rename(os.path.join(dir_, filename), os.path.join(dir_, filename.lower()))
        
    
   
df = pd.read_excel(home('~/Dropbox/Uni/DTC1/Insects/BL/datasets/dataset4-nounknown.xlsx'), parse_cols=(1,4) )
df = df[df.sound.str.lower().isin([x[0:-4] for x in available_files])]
df['species_latin'] = [re.sub("[\. ]", "", x.lower()) for x in df['species_latin']]
df['sound'] = [os.path.join(dir_, x.lower() + ".wav") for x in df['sound']]

### if you want a list of species
#df.groupby(['species_latin']).agg(len).sort('sound', ascending=False).to_html('/tmp/species.html')

### if you remove parse_cols from read_excel, you can find families as well
#df.groupby(['family_latin']).agg({'id' : len}).sort('id', ascending=False).to_html('/tmp/families.html')



print df.describe()
#print df.species_latin.unique(), len(df.species_latin.unique())

train_labels_csv = settings.modelname+"-train.csv"
df.to_csv(train_labels_csv, header=False, index=False)

settings.LABELS = {'default' : {
   'train' : train_labels_csv,
   'test'  : None
}}


from classispecies.classispecies import Classispecies
class BlOrthopteraModel(Classispecies):
    pass

for analyser in ["hertzfft"]: #, "mel-filterbank", "mel-fft", "mfcc"]: # "mel-filterbank", "mfcc",
    for classifier in ["decisiontree", "randomforest"]:
        for sec_segments in [5.0]:
            
            print "=" * 80
            print "ANALYSER:   %s" % analyser
            print "CLASSIFIER: %s" % classifier
            print "SEC SEGM:   %ss" % sec_segments
            
            settings.classifier = classifier
            settings.analyser   = analyser
            #settings.n_segments = 5
            #settings.sec_segments = sec_segments
            settings.sec_segments = sec_segments
            settings.NMFCCS = 13
            
            
            model = BlOrthopteraModel()
            model.run()
