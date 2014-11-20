'''
Copyright Davide Zilli 2014

:author:    Davide Zilli
:date:      07/10/2014

usage:
    python $0
    
Classify wildlife sounds.

This file:
* extracts MFCCs
* saves the data to a CSV file

The data is then (manually, at the moment) uploaded to Azure ML and classified
by a random forest classifier. 

'''

import os
import socket

import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.vq import vq, whiten, kmeans
import scipy.io.wavfile as wav

from pprint import pprint

from confusion import ConfusionMatrix
from features import mfcc
from features import logfbank

## settings
SPECTR_PLOT     = True
CONFUSION_PLOT  = True
PLOT            = SPECTR_PLOT or CONFUSION_PLOT
ROOTSOUNDPATHS = {"MSRC-3617038": "C:/Users/t-davizi/OneDrive/Work/SoundTrap/Resources",
              "DavsBook.local"    : "/Volumes/DATA/Cloud/OneDrive/Work/SoundTrap/Resources",
              "default"     : os.path.expanduser("~/Dropbox/Shared/xenocanto")}

SOUNDPATH = os.path.join(ROOTSOUNDPATHS.get(socket.gethostname(), ROOTSOUNDPATHS["default"]),
                         "sample")
CLASSES         = [ 
'chl',
'fri',
'bla',
'syl',
'car',
'cic',
'roe',
]
NCLASSES        = len(CLASSES)
NMFCCS          = 24 # number of MFCCs.
NFEATURES       = 1

import matplotlib as mpl
mpl.rcParams["font.family"] = "Segoe UI"
mpl.rcParams["font.size"] = "15"

def savedata(filename, obj):
    np.savetxt(filename, obj, delimiter=",", fmt="%s", comments = '',
               header= ",".join(["label"] +
                                #["mean%d" % x for x in range(NMFCCS-1)] +
                                #["var%d"  % x for x in range(NMFCCS-1)] +
                                ["max%d"  % x for x in range(NMFCCS-1)] ))
    
if __name__ == '__main__':
    
    #soundfiles = ["silence_1sec.wav", "sweep_1sec.wav"]
    soundfiles = [os.path.join(SOUNDPATH, file_) for file_ in os.listdir(SOUNDPATH)
                  if any(animal in file_ for animal in CLASSES)]

    soundfiles = []
    for root, subFolders, files in os.walk(SOUNDPATH):
        for file_ in files:
            if any(animal in file_ for animal in CLASSES):
                soundfiles.append(os.path.join(root, file_))
                
    basesoundfiles = [os.path.basename(x) for x in soundfiles]


    confusion = ConfusionMatrix(CLASSES)
    
    if SPECTR_PLOT:
        fig = plt.figure(figsize=(30,30))
        
    labels = np.array([os.path.basename(x)[0:3] for x in soundfiles]).reshape(len(soundfiles),1)
    #feature_dtype = [("label", 'S6')] +\
    #                [("mean%d" % x, 'f8') for x in range(NMFCCS-1)] +\
    #                [("var%d" % x, 'f8') for x in range(NMFCCS-1)] +\
    #                [("max%d" % x, 'f8') for x in range(NMFCCS-1)]
    feature_dtype = [("label", 'S6')] +\
                    [("max%d" % x, 'f8') for x in range(NMFCCS-1)] 
    #X = np.ndarray( (len(soundfiles)), dtype=feature_dtype)
    X = np.zeros( (len(soundfiles), (NMFCCS-1)*1) )
    #X = np.zeros( (len(soundfiles), (NMFCCS-1)) )
    #X.label = labels
    
    counter = 0
    
    for soundfile in soundfiles:
    
        print os.path.basename(soundfile)
        (rate,signal) = wav.read(soundfile)
        
        if len(signal.shape) > 1:
            signal = signal[:,0]
        signal = signal[:rate*20] # max 30 sec
        
        try:
            mfcc_feat = np.clip(np.nan_to_num(mfcc(signal, rate, numcep=NMFCCS)), a_min=-100, a_max=100)
        except MemoryError:
            print "MemoryError on", os.path.basename(soundfile)
            continue
    
        mean = np.nanmean(mfcc_feat[:,1:], axis=0) # exclude the first one
        #var  = np.nanvar(mfcc_feat[:,1:], axis=0)  # exclude the first one
        X[counter,:NMFCCS-1] = mean
        #X[counter,NMFCCS-1:(NMFCCS-1)*2]  = var
        #max_ = np.nanmax(mfcc_feat[:,1:], axis=0) # exclude the first one
        #X[counter,(NMFCCS-1)*2:] = max_
        #X[counter,:] = max_        
    
        counter += 1
    
        # plot spectrogram and MFCCs
        if SPECTR_PLOT:
            ax = fig.add_subplot(len(soundfiles),3,counter*3-2)
            specgram = ax.specgram(signal, Fs = rate, scale_by_freq=True, rasterized=True)
        
            ax.autoscale(tight=True)           # no space between plot an axes
            yticks = np.arange(0,rate/2,5000)    # |
            ax.set_yticks(yticks)              # |
            ax.set_yticklabels(yticks/1000)    # |
            ax.set_ylabel("Freq (kHz)")        # |> change Hz to kHz
            
            xticklabels = (["00:00", "00:05", "00:10", "00:15", "00:20", "02:30", "03:00", "03:30"])
            xticks = np.arange(0,(len(signal)/rate)+1, 5)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            if counter == len(soundfiles):
                ax.set_xlabel("Time (mm:ss)")
            
            
            ax = fig.add_subplot(len(soundfiles),3,counter*3-1)
            cax = ax.pcolormesh(np.transpose(mfcc_feat))
            ax.autoscale(tight=True)           # no space between plot an axes
            
            #plt.title(soundfile)
            fig.colorbar(cax)
        
    
    
    
    Y = np.hstack( (labels, X.astype('S8')) )
    
    # sanity check     
    assert X.shape[0] == Y.shape[0] == labels.shape[0] == len(soundfiles)
    assert X.shape[1] * NFEATURES +1 == Y.shape[1]
    
    savedata("training.csv" , Y[::2] ) # every even row in X
    savedata("comparing.csv", Y[1::2]) # every odd row in X
    
    X = whiten(np.array(X))
    
    codebook, distortion = kmeans(X,NCLASSES)
    code, dist = vq(X, codebook)
    
    ''' enforce the same sorting of k-means output classes and input ``CLASSES``.'''
    counter = 0
    d = {}
    for x_ in code:
        if x_ not in d:
            d[x_] = counter
            counter+= 1
    code = [d[x_] for x_ in code]
    
    matches = zip(basesoundfiles, code)
    
    stats = {k:{j:0 for j in CLASSES} for k in range(NCLASSES)}
    counter = 0
    for file_, class_ in matches:
        counter += 1
        animal = os.path.basename(file_)[:3]
        stats[class_][animal] += 1
        confusion.add(animal, CLASSES[class_])
        
        print "%s: class %d" % (animal, class_)
    
        if SPECTR_PLOT:
            ax = fig.add_subplot(len(soundfiles),3,counter*3-1)
            ax.text(4000, 6, "%10s: class %d" % (animal, class_))
    
    if SPECTR_PLOT:
        plt.savefig('classispecies.png')
        #plt.show()
        
    if CONFUSION_PLOT:
        confusion.plot()
        
    print stats
    print confusion
    
    plt.clf()
