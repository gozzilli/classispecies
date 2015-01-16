# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 11:53:19 2015

@author: Davide Zilli

Helper module that creates label data from a list of sound files
"""

import os
import numpy as np

from classispecies.utils import misc
from classispecies import settings
from sklearn.preprocessing import label_binarize

logger = misc.config_logging("makelabels")


def list_soundfiles():
    ''' Select files in soundpath '''

    def extract(set_):
        soundfiles = []
        for root, subFolders, files in os.walk(misc.get_soundpath(set_=set_)):
            logger.info ("%s files in %s" % (set_, root))

            for file_ in files:
                #if any(animal in file_ for animal in settings.CLASSES):
                    if file_.lower().endswith(".wav"):
                        soundfiles.append(os.path.join(root, file_))

        soundfiles = sorted(soundfiles)

        if not soundfiles:
            raise Exception("No sound file selected")

        return soundfiles

    train_soundfiles = extract("train")
    
    if misc.get_soundpath(set_="test"):
        test_soundfiles  = extract("test")
    
    else:
        test_soundfiles = None

    #basesoundfiles = [os.path.basename(x) for x in soundfiles]

    return np.array(train_soundfiles), np.array(test_soundfiles)


#def make_labels_from_filename(soundfiles):
def make_labels_from_list_of_files(train_soundfiles, 
                                   test_soundfiles=None, 
                                   modelname=settings.modelname, 
                                   is_multilabel=settings.MULTILABEL):    
    
    def make(soundfiles, outfilename):

        named_labels  = np.array([os.path.basename(x)[0:3] for x in soundfiles], dtype="S3")
        classes = np.unique(named_labels)
        bin_labels  = label_binarize(named_labels, classes)
        
        if is_multilabel:
            labels = np.hstack( (soundfiles[np.newaxis].T, bin_labels) )
        else:
            labels = np.vstack( (soundfiles, named_labels) ).T

        logger.info("saving labels to %s" % outfilename)
        np.savetxt(outfilename, labels, delimiter=",", fmt='"%s"'
                # fmt="%s", comments = '', header= ",".join(["label"] + ...)
                )
    
    out_trainlabels = "%s-train.csv" % modelname
    make(train_soundfiles, out_trainlabels)

    
    if test_soundfiles:
        out_testlabel = "%s-test.csv" % modelname
        make(train_soundfiles, out_testlabel)
    else:
        out_testlabel = None
    
    return out_trainlabels, out_testlabel



