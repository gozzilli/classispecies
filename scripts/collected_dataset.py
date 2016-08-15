'''
Created on 27 Jan 2015

@author: Davide Zilli
'''

from os.path import expanduser as h
import pandas as pd

WAV_DIR = '/home/dz2v07/cicada-largefiles/insects-collected/5sec/'
modelname = "collected"

df = pd.read_csv(h('~/Dropbox/Uni/DTC1/Insects/Collected/collected2.csv'), usecols=(2,0))
df.soundfile = WAV_DIR + df.soundfile + ".wav"
df.dutch_orth_id = ["%02d" % x for x in df.dutch_orth_id]

df = df[["soundfile", "dutch_orth_id"]]
df_train = df[df.soundfile.str.endswith("1.wav")]
df_test = df[df.soundfile.str.endswith("2.wav")]


df_train.to_csv(modelname + "-train.csv", header=False, index=False)
df_test.to_csv (modelname + "-test.csv",  header=False, index=False)


