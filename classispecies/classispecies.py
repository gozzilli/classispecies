# -*- coding: utf-8 -*-

from __future__ import print_function

from abc import ABCMeta
from collections import OrderedDict
from datetime import datetime, timedelta
import os
from pprint import pformat

from pandas.io.parsers import read_csv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored

from featextr import FeatureSet
import numpy as np
import pandas as pd
import settings
from utils import misc
from utils.confusion import ConfusionMatrix
from utils.plot import classif_plot, feature_plot

import traceback
import time

F32MIN = np.finfo('float32').min
F32MAX = np.finfo('float32').max



mel_feat = None
signal__ = None

max__ = 0
max__file = ""

IMPORTANT = "blue"
WARNING   = "red"
RESULT    = "green"
POSITIVE  = "green"
NEGATIVE  = "red"

# labels = None
# confusion = None

logger = misc.config_logging("classispecies")
logger2 = misc.config_logging("f1")
logger3 = misc.config_logging("confusion")

class Classispecies(object):
    __metaclass__ = ABCMeta
    
    rocauc           = None
    f1score          = None
    rocauc_combined  = None
    f1score_combined = None


    def get_classes(self):
        return self.classes

    def set_classes(self, classes):
        self.classes = classes

    def __init__(self, analyser=None, classifier=None, modelname=None,
                 highpass_cutoff=None, normalise=None, n_segments=None, sec_segments=None):
        
        
        self.train_soundfiles, self.train_labels, \
        self.test_soundfiles,  self.test_labels, self.classes \
            = self.load()

        self.missingTestLabels = not settings.SPLIT_TRAINING_SET and \
            (self.test_labels == None or np.all(np.unique(self.test_labels) == np.array(None)))
            
        self.classifier   = classifier   or settings.classifier
        self.analyser     = analyser     or settings.analyser
        self.modelname    = modelname    or settings.modelname
        self.highpass_cutoff = highpass_cutoff or settings.highpass_cutoff
        self.normalise    = normalise    or settings.normalise
        self.n_segments   = n_segments   or settings.n_segments
        self.sec_segments = sec_segments or settings.sec_segments
        
        an = "%s %s%s%s%s%s" % (self.classifier, "mel" if settings.extract_mel else "hertz",
                                                         " log" if settings.extract_dolog else "",
                                                         " dct" if settings.extract_dct else "",
                                                         " %s" % settings.agg,
                                                         " %.1fsec" % self.sec_segments if self.sec_segments else " entire")
        self.an = an
        logger.info ("INIT %s" % an)

        if self.n_segments and self.sec_segments:
            raise ValueError("`sec_segments` and `n_segments` are mutually exclusive parameters.")

        settings.CLASSES = self.classes

        self.results = {"analyser"      : self.analyser,
                        "classifier"    : self.classifier,
                        "n_train_files" : len(self.train_soundfiles),
                        "n_test_files"  : len(self.test_soundfiles) if isinstance(self.test_soundfiles, pd.DataFrame) else 0,
                        "n_chunks"      : "%s sec" % self.sec_segments if self.sec_segments else "%s" % self.n_segments
                        }

    def load(self, is_multilabel=None, usecols_train=None, usecols_test=None):
        ''' Load all sound files and labels. 
        
        The test set is optional, as it can be generated from splitting the train
        set later in the code. The training set is mandatory. 
        
        Params:
            is_multilabel:  whether the classification is multi-output or not, i.e.
                            if there can be more than one class present in each 
                            recording. Defaults to the value of settings.MULTILABEL
            usecols_train:  select specific column in the training set. Default is
                            to select all.
            usecols_test:   select specific column in the test set. Default is to 
                            select all.
                            
        Return:
            A tuple of:
            train_soundfiles
            train_labels
            test_soundfiles
            test_labels
            classes         
        '''
        
        is_multilabel = is_multilabel if is_multilabel != None else settings.MULTILABEL
    
        train_label_path = misc.get_path(settings.LABELS, "train")
        test_label_path  = misc.get_path(settings.LABELS, "test")
        
        ### TRAINING SET ###
        if not train_label_path or not os.path.exists(train_label_path):
            raise Exception("Cannot load train labels")

        
        labels = read_csv(train_label_path, header=None)
        #labels = np.array(train_label_path, delimiter=",", dtype="S", usecols=usecols_train)
        train_soundfiles = labels[0]
        train_labels     = labels.ix[:,1:].fillna(0).as_matrix()
        
        if not is_multilabel: train_labels = np.ravel(train_labels)
    
        
        ### TESTING SET ###
        if test_label_path and os.path.exists(test_label_path):
            #labels = np.loadtxt(test_label_path, delimiter=",", usecols=usecols_test)
            labels = read_csv(test_label_path, header=None)
            test_soundfiles = labels[0]
            test_labels     = labels.ix[:,1:].fillna(0).as_matrix()

            if test_labels == None:
                logger.info("Test files present, but no test labels found")
                self.missingTestLabels = True
                test_labels = None
    
            #assert np.all(np.unique(test_labels ) == np.array([0,1]))
            #assert len(classes) == test_labels.shape[1]
            
            if not is_multilabel: test_labels = np.ravel(test_labels)
    
    
        else:
            test_labels     = None
            test_soundfiles = None
        
        ### CLASSES ###        
        if is_multilabel:    
            
            classes = range(train_labels.shape[1])
            assert np.all(np.unique( train_labels ) == np.array([0,1]))
        
        else:
            classes = np.unique(train_labels).astype("S3")
    
        return train_soundfiles, train_labels, test_soundfiles, test_labels, classes

    def featxtr(self):

        params = [self.analyser, self.highpass_cutoff, self.normalise,
                  self.n_segments, self.sec_segments]
        cut = settings.test_size

        train_fs = FeatureSet("train", self.an)
        test_fs  = FeatureSet("test", self.an)
        

        if settings.SPLIT_METHOD == "split-before":

            if settings.SPLIT_TRAINING_SET:

                logger.warn (colored("Splitting training set for train/test data (%.2f) before extraction." % settings.test_size, WARNING))

                nsamples = len(self.train_soundfiles)
                np.random.seed(settings.RANDOM_SEED)
                indices = np.random.permutation(nsamples)
                idx_tr, idx_te = indices[:cut*nsamples], indices[cut*nsamples:]

                self.test_soundfiles  = self.train_soundfiles[idx_te]
                self.train_soundfiles = self.train_soundfiles[idx_tr]

                if len(self.train_labels.shape) == 2:
                    self.test_labels  = self.train_labels[idx_te,:]
                    self.train_labels = self.train_labels[idx_tr,:]
                elif len(self.train_labels.shape) == 1:
                    self.test_labels  = self.train_labels[idx_te]
                    self.train_labels = self.train_labels[idx_tr]
                else:
                    raise ValueError("Train labels can never have more than 2 dimensions")
                

                self.unchunked_test_labels = self.test_labels.copy()


            ### TRAINING SET ######################################################

            logger.info (colored ("[%s] analysing training files...\n" % settings.analyser, IMPORTANT))
            train_fs.extract(self.train_soundfiles, self.train_labels, *params)

            ### TE$T SET ##########################################################

            logger.info (colored ("[%s] analysing testing files..." % settings.analyser, IMPORTANT))
            test_fs.extract(self.test_soundfiles, self.test_labels, *params)



        elif settings.SPLIT_METHOD == "split-after":


            train_fs.extract(self.train_soundfiles, self.train_labels, *params)

            if settings.SPLIT_TRAINING_SET:

                logger.warn (colored("Splitting training set for train/test data (%.2f) after extraction." % settings.test_size, WARNING))

                train_fs, test_fs, self.train_soundfiles, self.test_soundfiles = train_fs.split(self.train_soundfiles, cut)

                u = OrderedDict()
                for soundfile, _, _, lab in test_fs.db:
                    u[soundfile] = lab
                self.unchunked_test_labels = np.array(u.values())
            else:
                logger.info (colored ("[%s] analysing testing files..." % settings.analyser, IMPORTANT))
                test_fs.extract(self.test_soundfiles, self.test_labels, *params)
                self.unchunked_test_labels = self.test_labels.copy()


        else:
            raise ValueError("Method %s for splitting training data unknown" )
        

        self.results.update({
            "min_sound_length_train" : train_fs.min_length,
            "max_sound_length_train" : train_fs.max_length,
            "avg_sound_length_train" : train_fs.avg_length,
            "n_train_files"          : len(train_fs),
            
            "min_sound_length_test"  : test_fs.min_length,
            "max_sound_length_test"  : test_fs.max_length,
            "avg_sound_length_test"  : test_fs.avg_length,
            "n_test_files"           : len(test_fs),

            "nsamples"  : train_fs.data.shape[0],
            "nfeatures" : train_fs.data.shape[1],
            })

        logger.debug( pformat(self.results) )
        logger.debug( "train truth: %d, test truth: %d" % (len(train_fs.db.truth), len(test_fs.db.truth)) )

        return train_fs, test_fs


    def export_for_azure(self, X, labels, n_soundfiles):

        temp_labels = labels.reshape(len(labels),1)
        Y = np.hstack( (temp_labels, X.astype('S8')) )

        # sanity check
        assert X.shape[0] == Y.shape[0] == temp_labels.shape[0] == n_soundfiles
        #assert X.shape[1] * settings.NFEATURES +1 == Y.shape[1]

        self.savedata("training" , Y[::2] ) # every even row in X
        self.savedata("comparing", Y[1::2]) # every odd row in X

        return Y


    #def classify(self, data_training, labels_training, data_testing, labels_testing):
    def classify(self, train_fs, test_fs):

        data_training, labels_training = train_fs.data, train_fs.db.truth
        data_testing,  labels_testing  = test_fs.data,  test_fs.db.truth 
        
        f1 = f1_merged = auc = auc_merged = np.nan
        

        if not settings.MULTILABEL:
            labels_training = np.ravel(labels_training)
            labels_testing  = np.ravel(labels_testing)


        if self.classifier == "decisiontree":
            clf = DecisionTreeClassifier(random_state=settings.RANDOM_SEED)
        elif self.classifier == "randomforest":
            clf = RandomForestClassifier(random_state=settings.RANDOM_SEED)
        else:
            raise Exception("Unknown classifier")

        if not np.all(np.isfinite(data_training)):
            data_training = np.clip(np.nan_to_num(data_training), F32MIN, F32MAX)
            
        if not np.all(np.isfinite(data_testing)):
            data_testing = np.clip(np.nan_to_num(data_testing), F32MIN, F32MAX)
                    
        clf = clf.fit(data_training, labels_training)
        self.clf = clf


        try:
            self.prediction = clf.predict(data_testing)#.reshape(len(data_testing),1)
        except ValueError:
            print (np.unique(data_testing))
            np.savetxt('/tmp/datatesting.txt', data_testing)
            raise
        self.predict_logproba = clf.predict_log_proba(data_testing)
        self.AAA = clf.predict_proba(data_testing)
        self.predict_proba = np.array(self.AAA)
        
#         print ("LABELS TESTING:\n", labels_testing, "uniq:", np.unique(labels_testing))
#         print ("DATA TRAINING shape:", data_training.shape)
#         print ("DATA TESTING shape:", data_testing.shape)
#         print ("LABELS TRAINING shape:", labels_training.shape)
#         print ("LABELS TESTING shape:", labels_testing.shape)
#         print ("PREDICT proba shape:", self.predict_proba.shape)
#         print ("PREDICT proba uniq:", np.unique(self.predict_proba))

        self.results["score"] = clf.score(data_testing, labels_testing)
        logger.info (colored("score: %.3f" % self.results["score"]), RESULT)
        
        

        if settings.MULTILABEL:
            try:
                self.predict_proba = self.predict_proba[:,:,1].T
            except IndexError:
                self.predict_proba = self.predict_proba
        else:
            self.predict_proba = self.predict_proba

        c = OrderedDict()
        for i in range(len(test_fs)):
            
            j = test_fs.db.soundfile[i]
            if j in c:
                c[j].append(self.predict_proba[i])
            else:
                
                c[j] = [self.predict_proba[i]]

        res_ = []
        for v in c.values():
            res_.append(np.mean(v, axis=0))
        self.c = np.array(c.values())
        self.res_ = np.array(res_)
        

        if not self.missingTestLabels:

            if settings.MULTILABEL:
                if settings.CLASSIF_PLOT: classif_plot(labels_testing, self.prediction)

                print ("SHAPES:", labels_testing.shape, self.predict_proba.shape)
                auc = roc_auc_score(labels_testing, self.predict_proba)
                

                tp = np.sum((labels_testing + self.prediction) == 2)
                fp = np.sum((labels_testing - self.prediction) == -1)
                fn = np.sum((labels_testing - self.prediction) == 1)
                logger.info ("TP: %d" % tp)
                logger.info ("FP: %d" % fp)
                logger.info ("FN: %d" % fn)
                self.results["auc"] = auc
                self.results["tps"] = tp
                self.results["fps"] = fp
                self.results["fns"] = fn
                
                auc_merged = roc_auc_score(self.unchunked_test_labels, self.res_)
                self.results["roc_merged" ] = auc_merged
                
                e = np.array([[1 if x == max(y) and x > 0 else 0 for x in y] for y in self.res_])
                f = self.unchunked_test_labels
                
                f1_merged = f1_score(f, e)
                self.results["f1_merged"] = f1_merged
                

            else: # single label
                confusion = ConfusionMatrix(self.classes)
                res_colored = ""
                self.truepred = zip(labels_testing, self.prediction)
                for true, pred in zip(labels_testing, self.prediction):

                    logger.debug (colored("true %s, predicted %s" % (true, pred), POSITIVE if true == pred else NEGATIVE))
                    res_colored += "<span style='color: %s;'>true %s, predicted %s</span><br>" % (POSITIVE if true == pred else NEGATIVE, true, pred)
                    confusion.add(true, pred)

                self.results["colored_res"] = res_colored
                if settings.CONFUSION_PLOT:
                    logger.debug (confusion)
                    confusion.plot(outputname=misc.make_output_filename("confusion", "", settings.modelname, settings.MPL_FORMAT))
                self.results["confusion_matrix"] = confusion.html()
                self.confusion = confusion
                
                
                
                ##### NEW 
                train_bin_labels = label_binarize(self.train_fs.db.truth, self.classes)
                test_bin_labels = label_binarize(self.test_fs.db.truth, self.classes)
                self.test_bin_labels = test_bin_labels
                self.train_bin_labels = train_bin_labels
                
                if 0 in np.sum(test_bin_labels, axis=0):
                    logger.warn(colored("Zeros in truth. roc score not defined.", WARNING))

                else:
                
                    if self.classifier == "decisiontree":
                        clf2 = DecisionTreeClassifier(random_state=settings.RANDOM_SEED)
                    elif self.classifier == "randomforest":
                        clf2 = RandomForestClassifier(random_state=settings.RANDOM_SEED)
                    else:
                        raise Exception("Unknown classifier")
                    clf2.fit(data_training, train_bin_labels)
                     
                    pred       = clf2.predict(data_testing)#.reshape(len(data_testing),1)
                    try:
                        pred_proba = np.array(clf2.predict_proba(data_testing))[:,:,1].T
                    except IndexError:
                        pred_proba = np.array(clf2.predict_proba(data_testing))                
                    self.pred = pred
     
                    self.pred_proba = pred_proba 
                    auc = roc_auc_score(test_bin_labels, self.predict_proba)
                    
#                     tp = np.sum((test_bin_labels + self.prediction) == 2)
#                     fp = np.sum((test_bin_labels - self.prediction) == -1)
#                     fn = np.sum((test_bin_labels - self.prediction) == 1)
#                     logger.info ("TP: %d" % tp)
#                     logger.info ("FP: %d" % fp)
#                     logger.info ("FN: %d" % fn)
                    
#                     self.results["tps"] = tp
#                     self.results["fps"] = fp
#                     self.results["fns"] = fn
                    
                    bin_labels = label_binarize(self.unchunked_test_labels, self.classes)
                    self.bin_labels = bin_labels
                    #auc_merged = roc_auc_score(bin_labels, self.res_)
                    
                    
                    e = np.array([self.classes[x] for x in np.argmax(self.res_, axis=1)])
                    f = self.unchunked_test_labels.astype("S2")
                    
                    f1_merged = f1_score(f, e, average=None if len(np.unique(f)) == 2 else "weighted")
                
                self.results["auc"] = auc
                self.results["f1_merged"] = f1_merged
                self.results["roc_merged" ] = auc_merged


            res = np.ravel(labels_testing) == np.ravel(self.prediction)
            self.res = res

            f1  = f1_score(labels_testing, self.prediction, 
                           average=None if len(np.unique(labels_testing)) == 2 else "weighted")
            self.results["f1"]  = f1

            if settings.MULTILABEL:
                total = np.sum(labels_testing)

                self.results["correct"] = tp
                self.results["total"]   = total
                self.results["correct_percent"] = float(tp)/total*100

#                print ("correct %d/%d (%.3f%%)" % (self.results["correct"], self.results["total"], self.results["correct_percent"]))
#                print ("correct %d/%d (%.3f%%)" % (tp, int(np.sum(self.test_labels)), tp/np.sum(self.test_labels)*100))

            else:
                total = len(labels_testing)
                self.results["correct"] = np.sum(res)
                self.results["total"]   = np.size(res)
                self.results["correct_percent"] = np.sum(res)/float(np.size(res))*100

            
            logger.info (self.an)
            if settings.MODEL_BINARY:
                logger.info (colored("F1 score: %s"  % f1, RESULT))
            else:
                logger.info (colored("F1 score: %s %s" % ("%.3f" % f1  if f1  else "n/a",  "(merged: %s)" % ("%.3f" % f1_merged  if f1_merged  else "n/a")), RESULT))
                logger.info (colored(" ROC AUC: %s %s" % ("%.3f" % auc if auc else "n/a", "(merged: %s)" % ("%.3f" % auc_merged if auc_merged else "n/a")), RESULT))
                logger.info ("correct %d/%d (%.3f%%)" % (self.results["correct"], self.results["total"], self.results["correct_percent"]))
                

            
            
            
                
            

    def run(self):
        '''
        Execute an example of the module usage.

        * extract features for all appropriate files in soundpath
        * classify locally
        * print if it was correctly classified

        '''

        logger.info  (colored ("\nstarting %s" % type(self).__name__, IMPORTANT))
        logger.debug ("classes are: %s" % settings.CLASSES)



        ### FEATURE EXTRACTION ################################################

        logger.info (colored ("[%s] analysing..." % settings.analyser, IMPORTANT))
        train_fs, test_fs = self.featxtr()
        self.train_fs = train_fs
        self.test_fs  = test_fs
        
        ### PRE-PROCESSING ####################################################

        logger.info (colored ("preprocessing...", IMPORTANT))
        train_fs.preprocess()
        test_fs.preprocess()
        
        if settings.FEATURES_PLOT:
            feature_plot(train_fs.data, settings.modelname, settings.MPL_FORMAT)

        if not self.missingTestLabels:
            if settings.MULTILABEL:
                self.results["items_to_classify"] = np.sum(test_fs.db.truth)
            else:
                self.results["items_to_classify"] = len(test_fs.data)


        # CODE HAS CHANGED. EXPORT FOR AZURE NEEDS ADJUSTING AND TESTING
        #logger.info (colored ("exporting data...", "red"))
        #Y = self.export_for_azure(self.X, self.labels, len(self))

        ### CLASSIFICATION ####################################################

        logger.info (colored ("[%s] classifying..." % settings.classifier, IMPORTANT))
        if settings.classifier == 'decisiontree' or settings.classifier == 'randomforest':

            self.classify(train_fs, test_fs)

        elif settings.classifier == "kmeans":
            self.cluster(test_fs.data)
        
        else:
            raise Exception("Classifier not implemented.")

        if settings.DUMP_REPORT:
            misc.dump_report(self.results, self.modelname)
        
        return self.results
    
    def save_to_db(self):
        client = MongoClient()
        db = client.classispecies ## this is the database
        res = db[self.modelname]          ## this is the collection
        
        ## this is the collection
        doc = settings.get_all(as_string=False)
        r = self.results.copy()
        r.pop("settings", None), r.pop("colored_res", None), r.pop("confusion_matrix", None)
        
        doc.update(r)
        doc["timestamp"] = unicode(datetime.now().strftime("%d %b %Y, %H:%M:%S"))
        
        res.insert(doc)
        
        
def multirunner(Model, sec_segments_array=[5.0],
                iters = 100,
                run_key=misc.key_gen(5)):
    
    client = MongoClient()
    db = client.classispecies ## this is the database

    settings.DUMP_REPORT = False
    settings.FEATURES_PLOT = False
    settings.FEATURE_ONEFILE_PLOT = False
    #settings.RANDOM_SEED = None
    
    analysers   = ["multiple"]
    classifiers = ["decisiontree", "randomforest"]
    mels        = [False, True]
    logs        = [False, True]
    dcts        = [False, True]
    aggs        = [u"logmod", "mod", "mod0", u"μ+σ", u"μ", "max",]
    downscale_factors = [settings.downscale_factor]
    
    errors = []    
    
    counter = 1
    total   = len(sec_segments_array)*len(analysers)*len(classifiers)*len(logs)*len(mels)*len(dcts)*len(aggs)*iters
    for sec_segments in sec_segments_array:
        for analyser in analysers: # "mel-filterbank", "mfcc",
            for classifier in classifiers:
                for mel_ in mels:
                    for log_ in logs: 
                        for dct_ in dcts:
                            for agg_ in aggs:
                                                                
                                fft2_ = agg_ in ["mod", "mod0", "logmod"]
                                
                                for ds_ in downscale_factors:
                                    
                                    ds_ = ds_ if agg_ in ["mod"] else None
                                    
                                    settings.FORCE_FEATXTR    = False
                                    settings.FORCE_FEATXTRALL = True
                                 
                                    try:
                                        print ("=" * 80)
                                        print ("[%d/%d]" % (counter, total))
                                        an = "%s%s%s%s%s%s" % ("mel" if mel_ else "hertz",
                                                             " log" if log_ else "",
                                                             " dct" if dct_ else "",
                                                             " %s" % agg_ if agg_ else " no agg",
                                                             " %.1fsec" % sec_segments if sec_segments else " entire",
                                                             " %dds" % ds_ if ds_ else "")
                                        
                                        if not settings.FORCE_MULTIRUNNER_RECOMPUTE and \
                                            db[settings.modelname].find({
                                            "extract_mel"      : mel_,
                                            "extract_dolog"    : log_,
                                            "extract_dct"      : dct_,
                                            "classifier"       : classifier,
                                            "analyser"         : analyser,
                                            "sec_segments"     : sec_segments,
                                            "downscale_factor" : ds_,
                                            "agg"              : agg_.encode('utf8'),
                                                                }).count() > 0:
                                            logger.info(colored("Skipping %s" % an, WARNING))
                                            counter += iters
                                            continue
                                        else:
                                            pass
                                        
                                        print ("ANALYSER   %s" % an)
                                        print ("CLASSIFIER %s" % classifier)
                                        print ("SEC SEGM   %ss" % sec_segments)
                                        print ("MEL        %s" % mel_)
                                        print ("LOG        %s" % log_)
                                        print ("DCT        %s" % dct_)
                                        print ("MOD        %s" % fft2_)
                                        print (u"AGG        %s" % agg_)
                                        print ("DOWNSCALE  %s" % ds_)
                                         
                                        settings.sec_segments  = sec_segments
                                        settings.classifier    = classifier
                                        settings.analyser      = analyser
                                        settings.extract_mel   = mel_
                                        settings.extract_dolog = log_
                                        settings.extract_dct   = dct_
                                        settings.extract_fft2  = fft2_
                                        settings.agg           = agg_
                                        settings.downscale_factor = ds_
                                         
                                        if agg_ == u"μ+σ":
                                            settings.extract_mean   = settings.extract_std = True
                                            settings.extract_max    = False
                                            settings.extract_fft2   = False
                                            settings.extract_logmod = False
                                            settings.MOD_TAKE1BIN   = False

                                        elif agg_ == u"μ":
                                            settings.extract_mean   = True
                                            settings.extract_max    = settings.extract_std = False
                                            settings.extract_fft2   = False
                                            settings.extract_logmod = False
                                            settings.MOD_TAKE1BIN   = False
                                        
                                        elif agg_ == "max":
                                            settings.extract_mean   = settings.extract_std = False
                                            settings.extract_max    = True
                                            settings.extract_fft2   = False
                                            settings.extract_logmod = False
                                            settings.MOD_TAKE1BIN   = False
                                        
                                        elif agg_ == "mod":
                                            settings.extract_mean   = settings.extract_std = settings.extract_max = False
                                            settings.extract_fft2   = True
                                            settings.extract_logmod = False
                                            settings.MOD_TAKE1BIN   = False
                                        
                                        elif agg_ == "logmod":
                                            settings.extract_mean   = settings.extract_std = settings.extract_max = False
                                            settings.extract_fft2   = True
                                            settings.extract_logmod = True
                                            settings.MOD_TAKE1BIN   = False
                                        
                                        elif agg_ == "mod0":
                                            settings.extract_mean   = settings.extract_std = settings.extract_max = False
                                            settings.extract_fft2   = True
                                            settings.extract_logmod = False
                                            settings.MOD_TAKE1BIN   = True
                                        
                                        else:
                                            raise ValueError(u"Not a valid aggregator: %s" % agg_)
        
                                        start = time.time()

                                        f1s, rocs, f1m, rocm = [], [], [], []
                                        truepred = []
                                        for i in range(iters):
                                            
                                            #settings.FEATURE_ONEFILE_PLOT = counter == 1
                                            
                                            if agg_ == "mod":
                                                assert settings.extract_mean == settings.extract_std == settings.extract_max == False
                                            
                                            try: 
                                                logger.info (colored( "%s [%d/%d] %s %d/%d" %(run_key, counter, total, an, i+1, iters), IMPORTANT))
                                                model = Model()
                                                results = model.run()
                                                f1s.append (results["f1"])
                                                rocs.append(results["auc"])
                                                f1m.append (results["f1_merged" ])
                                                rocm.append(results["roc_merged"])
                                                truepred.append(model.truepred)
                                                
                                                settings.FORCE_FEATXTR    = False
                                                settings.FORCE_FEATXTRALL = True
                                            except ValueError:
                                                traceback.print_exc()
                                                errors.append("%d %s %s" % (counter, an, settings.modelname))
                                                raise
                                                
                                            except IndexError as e:
                                                traceback.print_exc()
                                                errors.append("%d %s %s" % (counter, an, settings.modelname))
                                                raise
                                            
                                            counter += 1
                                            
                                        results["f1"]    = np.mean(f1s)
                                        results["f1std"] = np.std(f1s)
                                        results["f1sem"] = np.sqrt(np.var(f1s)/len(f1s))
                                        print (rocs)
                                        results["auc"]   = np.nanmean(rocs)
                                        
                                        results["roc_merged_std"] = np.nanstd(rocs)
                                        results["rocsem"] = np.sqrt(np.nanvar(rocs)/len([x for x in rocs if x]))
                                        
                                        results["f1m"] = f1m
                                        results["rocm"]= rocm
                                        
                                        results["run_key"] = run_key
                                        
                                        logger2.info ("f1s are:\n%s", str(f1s))
                                        logger.info ("f1 mean: %.3f, std: %.3f" % (results["f1"], results["f1std"]))
                                        logger.info ("%d errors so far" % len(errors))
                                        
                                        confusion = ConfusionMatrix(model.classes)
 
                                        for t in truepred:
                                            for true, pred in t:
                                                confusion.add(true, pred)
                                        
                                        logger3.info("\n%s" %an)         
                                        logger3.info(str(confusion))
                                        logger3.info(confusion.toval())
                                        logger3.info("========================")
                                        
                                        elapsed = (time.time() - start)
                                        time_for_one = elapsed/float(total)/float(iters)
                                        print ("takes %d seconds to do %d iterations" % (int(time_for_one), iters))
                                        eta = str(timedelta(seconds=int((int(total) - iters*int(counter))*time_for_one)))
                                        print (eta)

                                        model.results = results
                                        if settings.SAVE_TO_DB:
                                            model.save_to_db()
                                    except ValueError as e:
                                        errors.append("%d %s %s %s" % (counter, an, settings.modelname, e))
                                        print (unicode(e).encode('utf8'))
                                        #return model
                                        raise
                                        traceback.print_exc()
                                        
                                    except Exception as e:
                                        errors.append("%d %s %s" % (counter, an, settings.modelname))
                                        print ("Major error in %d %s %s" % (counter, an, settings.modelname))
                                        print (e)
                                        traceback.print_exc()
                                        #return model
                                        raise
                                    
    print ("ALL ERRORS:\n", errors)
    return model


        
def multiextracter(Model, sec_segments_array=[5.0],
                iters = 1,
                run_key=misc.key_gen(5)):
    
    settings.DUMP_REPORT = False
    settings.FEATURES_PLOT = False
    settings.FEATURE_ONEFILE_PLOT = False
    settings.RANDOM_SEED = None
    
    analysers   = ["multiple"]
    classifiers = ["decisiontree", "randomforest"]
    mels        = [False, True]
    logs        = [False, True]
    dcts        = [False, True]
    aggs        = ["mod", u"μ+σ", "max"]
    
    errors = []    

    counter = 1
    total   = len(sec_segments_array)*len(analysers)*len(classifiers)*len(logs)*len(mels)*len(dcts)*len(aggs)*iters
    for sec_segments in sec_segments_array:
        for analyser in analysers: # "mel-filterbank", "mfcc",
            for classifier in classifiers:
                for mel_ in mels:
                    for log_ in logs: 
                        for dct_ in dcts:
                            for agg_ in aggs:
                                                                
                                fft2_ = agg_ == "mod"
                                
                                if fft2_:
                                    downscale_factors_ = [64]
                                else:
                                    downscale_factors_ = [None]
                                    
                                for ds_ in downscale_factors_:
                                    
                                    settings.FORCE_FEATXTR    = True
                                    settings.FORCE_FEATXTRALL = True
                                 
                                    print ("=" * 80)
                                    print ("[%d/%d]" % (counter, total))
                                    an = "%s%s%s%s%s%s" % ("mel" if mel_ else "hertz",
                                                         " log" if log_ else "",
                                                         " dct" if dct_ else "",
                                                         " %s" % agg_ if agg_ else " no agg",
                                                         " %.1fsec" % sec_segments if sec_segments else " entire",
                                                         " %dds" % ds_ if ds_ else "")
                                    
                                    print ((u"ANALYSER   %s" % an).encode('utf8'))
                                    print ("CLASSIFIER %s" % classifier)
                                    print ("SEC SEGM   %s" % sec_segments)
                                    print ("MEL        %s" % mel_)
                                    print ("LOG        %s" % log_)
                                    print ("DCT        %s" % dct_)
                                    print ("MOD        %s" % fft2_)
                                    print ((u"AGG        %s" % agg_).encode('utf8'))
                                    print ("DOWNSCALE  %s" % ds_)
                                     
                                    settings.sec_segments  = sec_segments
                                    settings.classifier    = classifier
                                    settings.analyser      = analyser
                                    settings.extract_mel   = mel_
                                    settings.extract_dolog = log_
                                    settings.extract_dct   = dct_
                                    settings.extract_fft2  = fft2_
                                    settings.agg           = agg_
                                    settings.downscale_factor = ds_
                                     
                                    if agg_ == u"μ+σ":
                                        settings.extract_mean = settings.extract_std = True
                                        settings.extract_max = False
                                    elif agg_ == "max":
                                        settings.extract_mean = settings.extract_std = False
                                        settings.extract_max = True
                                    elif agg_ == "mod":
                                        settings.extract_mean = settings.extract_std = settings.extract_max = False
                                        settings.extract_fft2 = True
                                    else:
                                        raise ValueError("Not a valid aggregator")
    
                                    for i in range(iters):
                                        
                                        if agg_ == "mod":
                                            assert settings.extract_mean == settings.extract_std == settings.extract_max == False
                                        
                                        logger.info (colored( "%s [%d/%d] %s %d/%d" %(run_key, counter, total, an, i+1, iters), IMPORTANT))
                                        model = Model()
                                        try:
                                            model.featxtr()
                                        except TypeError:
                                            pass
                                        except ValueError:
                                            pass
                                        
                                        counter += 1
