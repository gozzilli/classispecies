# -*- coding: utf-8 -*-
"""
Create the CSV file for the BL Orthoptera data set.

Created on Sat Jan 24 20:01:18 2015

@author: Davide Zilli
"""

import os
from os.path import expanduser as home
import re
import pandas as pd

modelname  = "blorthoptera"
dir_ = home('~/cicada-largefiles/bl-files/all-16bit')
available_files = os.listdir(dir_)

FILES2LOWER = False   

if FILES2LOWER:
    
    for pdffilename in available_files:
        os.rename(os.path.join(dir_, pdffilename), os.path.join(dir_, pdffilename.lower()))
        
df = pd.read_excel(home('~/Dropbox/Uni/DTC1/Insects/BL/datasets/dataset4-nounknown.xlsx'), parse_cols=(1,4) )
df = df[df.sound.str.lower().isin([x[0:-4] for x in available_files])]
df['species_latin'] = [re.sub("[\. ]", "", x.lower()) for x in df['species_latin']]
df['sound'] = [os.path.join(dir_, x.lower() + ".wav") for x in df['sound']]

### if you want a list of species
#df.groupby(['species_latin']).agg(len).sort('sound', ascending=False).to_html('/tmp/species.html')

### if you remove parse_cols from read_excel, you can find families as well
#df.groupby(['family_latin']).agg({'id' : len}).sort('id', ascending=False).to_html('/tmp/families.html')


print df.describe()

train_labels_csv = modelname+"-train.csv"
df.to_csv(train_labels_csv, header=False, index=False)