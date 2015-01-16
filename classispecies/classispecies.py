# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from termcolor import colored
from abc import ABCMeta
from collections import OrderedDict
from pprint import pprint
from pandas.io.parsers import read_csv


import settings


import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

from featextr import FeatureSet
from utils.confusion import ConfusionMatrix
from utils.plot import classif_plot, feature_plot
from utils import misc

mel_feat = None
signal__ = None

max__ = 0
max__file = ""

IMPORTANT = "blue"
WARNING   = "red"
RESULT    = "green"
POSITIVE  = "green"
NEGATIVE  = "red"

labels = None
confusion = None

logger = misc.config_logging("classispecies")


class Classispecies(object):
    __metaclass__ = ABCMeta


    def get_classes(self):
        return self.classes

    def set_classes(self, classes):
        self.classes = classes

    def __init__(self, analyser=None, classifier=None, modelname=None,
                 highpass_cutoff=None, normalise=None, n_segments=None, sec_segments=None):

        self.train_soundfiles, self.train_labels, \
        self.test_soundfiles,  self.test_labels, self.classes \
            = self.load()

        #self.train_labels, self.test_labels, self.classes = self.load_labels()

        self.missingTestLabels = not settings.SPLIT_TRAINING_SET and \
            (self.test_labels == None or np.all(np.unique(self.test_labels) == np.array(None)))

        #self.nfeatures    = nfeatures    or settings.NFEATURES
        self.classifier   = classifier   or settings.classifier
        self.analyser     = analyser     or settings.analyser
        self.modelname    = modelname    or settings.modelname
        self.highpass_cutoff = highpass_cutoff or settings.highpass_cutoff
        self.normalise    = normalise    or settings.normalise
        self.n_segments   = n_segments   or settings.n_segments
        self.sec_segments = sec_segments or settings.sec_segments

        if self.n_segments and self.sec_segments:
            raise ValueError("`sec_segments` and `n_segments` are mutually exclusive parameters.")

        settings.CLASSES = self.classes

        self.results = {"analyser"      : self.analyser,
                        "classifier"    : self.classifier,
                        "n_train_files" : len(self.train_soundfiles),
                        "n_test_files"  : len(self.test_soundfiles) if self.test_soundfiles != None else 0,
                        "n_chunks"      : "%s sec" % self.sec_segments if self.sec_segments else "%s" % self.n_segments
                        }

    def load(self, is_multilabel=settings.MULTILABEL, usecols_train=None, usecols_test=None):
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
    
        train_label_path = misc.get_path(settings.LABELS, "train")
        test_label_path  = misc.get_path(settings.LABELS, "test")
        
        ### TRAINING SET ###
        if not train_label_path or not os.path.exists(train_label_path):
            raise Exception("Cannot load train labels")

        
        labels = read_csv(train_label_path, header=None, dtype="S")
        #labels = np.array(train_label_path, delimiter=",", dtype="S", usecols=usecols_train)
        train_soundfiles = labels[0]
        train_labels     = labels.ix[:,1:].as_matrix()
        
        if not is_multilabel: train_labels = np.ravel(train_labels)
    
        
        ### TESTING SET ###
        if test_label_path and os.path.exists(test_label_path):
            #labels = np.loadtxt(test_label_path, delimiter=",", usecols=usecols_test)
            labels = read_csv(test_label_path, header=None)
            test_soundfiles = labels[0]
            test_labels     = labels[:,1:].as_matrix()
    
            assert np.all(np.unique(test_labels ) == np.array([0,1]))
            #assert len(classes) == test_labels.shape[1]
    
    
        else:
            test_labels     = None
            test_soundfiles = None
            
    
        ### CLASSES ###        
        if is_multilabel:    
        
            classes = range(train_labels.shape[1])
            
            assert np.all(np.unique( train_labels ) == np.array([0,1]))
        
        else:
            classes = np.unique(train_labels) 
    
        return train_soundfiles, train_labels, test_soundfiles, test_labels, classes

    def featxtr(self):

        params = [self.analyser, self.highpass_cutoff, self.normalise,
                  self.n_segments, self.sec_segments]
        cut = settings.test_size

        train_fs = FeatureSet()
        test_fs  = FeatureSet()


        if settings.SPLIT_METHOD == "split-before":

            if settings.SPLIT_TRAINING_SET:

                logger.warn (colored("Splitting training set for train/test data (%.2f) before extraction." % settings.test_size, WARNING))

                nsamples = len(self.train_soundfiles)
                indices = np.random.permutation(nsamples)
                idx_tr, idx_te = indices[:cut*nsamples], indices[cut*nsamples:]

                self.test_soundfiles  = self.train_soundfiles[idx_te]
                self.train_soundfiles = self.train_soundfiles[idx_tr]

                self.test_labels  = self.train_labels[idx_te,:]
                self.train_labels = self.train_labels[idx_tr,:]

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
                print ("len: utl:", self.unchunked_test_labels.shape)
            else:
                logger.info (colored ("[%s] analysing testing files..." % settings.analyser, IMPORTANT))
                test_fs = FeatureSet()
                test_fs.extract(self.test_soundfiles, self.test_labels, *params)


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

        pprint (self.results)
        print (len(train_fs.db.truth), len(test_fs.db.truth))


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

        if not settings.MULTILABEL:
            labels_training = np.ravel(labels_training)
            labels_testing  = np.ravel(labels_testing)


        if self.classifier == "decisiontree":
            clf = DecisionTreeClassifier()
        elif self.classifier == "randomforest":
            clf = RandomForestClassifier()
        else:
            raise Exception("Unknown classifier")


        clf = clf.fit(data_training, labels_training)


        self.prediction = clf.predict(data_testing)#.reshape(len(data_testing),1)
        self.predict_logproba = clf.predict_log_proba(data_testing)
        self.predict_proba = np.array(clf.predict_proba(data_testing))

        if settings.MULTILABEL:
            self.predict_proba = self.predict_proba[:,:,1].T
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


                auc = roc_auc_score(labels_testing, self.predict_proba)

                tp = np.sum((labels_testing + self.prediction) == 2)
                fp = np.sum((labels_testing - self.prediction) == -1)
                fn = np.sum((labels_testing - self.prediction) == 1)
                logger.info (colored("ROC AUC: %.3f" % auc, RESULT))
                logger.info ("TP: %d" % tp)
                logger.info ("FP: %d" % fp)
                logger.info ("FN: %d" % fn)
                self.results["auc"] = auc
                self.results["tps"] = tp
                self.results["fps"] = fp
                self.results["fns"] = fn

            else: # single label
                confusion = ConfusionMatrix(self.classes)
                res_colored = ""
                for true, pred in zip(labels_testing, self.prediction):

                    logger.debug (colored("true %s, predicted %s" % (true, pred), POSITIVE if true == pred else NEGATIVE))
                    res_colored += "<span style='color: %s;'>true %s, predicted %s</span><br>" % (POSITIVE if true == pred else NEGATIVE, true, pred)
                    confusion.add(str(true), str(pred))

                self.results["colored_res"] = res_colored
                if settings.CONFUSION_PLOT:
                    logger.debug (confusion)
                    confusion.plot(outputname=misc.make_output_filename("confusion", "", settings.modelname, settings.MPL_FORMAT))
                self.results["confusion_matrix"] = confusion.html()
                self.confusion = confusion


            res = np.ravel(labels_testing) == np.ravel(self.prediction)
            self.res = res

            f1  = f1_score(labels_testing, self.prediction)
            self.results["f1"]  = f1
            logger.info (colored("f1: %.3f" % f1, RESULT))

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



            logger.info ("correct %d/%d (%.3f%%)" % (self.results["correct"], self.results["total"], self.results["correct_percent"]))

            if settings.MULTILABEL:
                logger.info (colored("ROC AUC  merged: %.3f" % roc_auc_score(self.unchunked_test_labels, self.res_), RESULT))
            
            # TODO: add F1 for merged.
            #logger.info (colored("F1 score merged: %.3f" % f1_score(self.unchunked_test_labels, self.res_), RESULT))



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

        misc.dump_report(self.results, self.modelname)