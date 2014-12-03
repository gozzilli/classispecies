# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys
from termcolor import colored
from abc import ABCMeta


import settings


import numpy as np
import scipy.io.wavfile as wav
from multiprocessing import Pool, Manager, cpu_count, util as m_util
from multiprocessing.pool import ApplyResult

from scipy.cluster.vq import vq, whiten, kmeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split

from featextr import exec_featextr
from utils.confusion import ConfusionMatrix
from utils.plot import classif_plot, feature_plot
from utils import misc

mel_feat = None
signal__ = None

max__ = 0
max__file = ""


class Classispecies(object):
    __metaclass__ = ABCMeta
    
        
    def get_classes(self):
        return self.classes
    
    def set_classes(self, classes):
        self.classes = classes
    
    def __init__(self, analyser=None, classifier=None, modelname=None,
                 highpass_cutoff=None, normalise=None, n_segments=None, sec_segments=None):

        self.train_soundfiles, self.test_soundfiles = self.list_soundfiles()
        
        self.train_labels, self.test_labels, self.classes = self.load_labels()
        
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
                        "n_test_files"  : len(self.test_soundfiles) if self.test_soundfiles else 0,
                        "n_chunks"      : "%s sec" % self.sec_segments if self.sec_segments else "%s" % self.n_segments
                        }
        
    def make_labels_from_filename(self, soundfiles):
            
        named_labels  = np.array([os.path.basename(x)[0:3] for x in soundfiles], dtype="S3")
        classes = np.unique(named_labels)
        bin_labels  = label_binarize(named_labels, classes)
        return bin_labels, classes, named_labels
                
        
        
    def load_labels(self, min_row=0, max_row=None, min_col=0, max_col=None):

        train_label_path = misc.get_path(settings.LABELS, "train")
        test_label_path  = misc.get_path(settings.LABELS, "test")
        
        print(settings.MULTILABEL , os.path.exists(train_label_path) , train_label_path)
        print(settings.MULTILABEL and os.path.exists(train_label_path) and train_label_path)
        if settings.MULTILABEL and os.path.exists(train_label_path) and train_label_path:
            train_labels = np.loadtxt(train_label_path, delimiter=",")
            if max_row == None: max_row = train_labels.shape[0]
            if max_col == None: max_col = train_labels.shape[1]
            
            train_labels = train_labels[min_row:max_row,min_col:max_col]
            classes  =  range(train_labels.shape[1])

            if os.path.exists(test_label_path):
                test_labels = np.loadtxt(test_label_path, delimiter=",")
                if max_row == None: max_row = test_labels.shape[0]
                if max_col == None: max_col = test_labels.shape[1]
                test_labels = test_labels[min_row:max_row,min_col:max_col]
            else:
                test_labels = None
                
            assert np.all(np.unique(train_labels ) == np.array([0,1]))
            if test_labels != None:
                assert np.all(np.unique(test_labels ) == np.array([0,1]))
                assert len(classes) == test_labels.shape[1]
        
        else:
            _, classes, train_labels = self.make_labels_from_filename(self.train_soundfiles)
            if self.test_soundfiles:
                _, tclasses, test_labels = self.make_labels_from_filename(self.test_soundfiles)
            else:
                test_labels = None
            
        
        return train_labels, test_labels, classes
    
    def list_soundfiles(self):
        ''' select files in soundpath that contain the names is CLASSES '''
        
        def extract(set_):
            soundfiles = []
            print (misc.get_soundpath(set_=set_))
            for root, subFolders, files in os.walk(misc.get_soundpath(set_=set_)):
                print ("selecting files in", root)
                
                for file_ in files:
                    #if any(animal in file_ for animal in settings.CLASSES):
                        if file_.lower().endswith(".wav"):
                            soundfiles.append(os.path.join(root, file_))
                
            soundfiles = sorted(soundfiles)
        
            if not soundfiles:
                raise Exception("No sound file selected")
            
            return soundfiles
        
        train_soundfiles = extract("train")            
        if False: # settings.SPLIT_TRAINING_SET:
            test_soundfiles  = train_soundfiles[1::2]
            train_soundfiles = train_soundfiles[0::2]
        else:
            test_soundfiles  = extract("test")
            
        #basesoundfiles = [os.path.basename(x) for x in soundfiles]
        
        return train_soundfiles, test_soundfiles
    
    
    def savedata(self, filename, obj):
        ''' export feature data to file so that they can be processed by Azure ML '''
        
        np.savetxt(misc.make_output_filename(filename, "", settings.modelname, "csv"), obj, delimiter=",",
                   fmt="%s", comments = '', header= ",".join(["label"] +
                                    #["mean%d" % x for x in range(settings.NMFCCS-1)] +
                                    #["std%d"  % x for x in range(settings.NMFCCS-1)] +
                                    ["max%d"  % x for x in range(settings.NMFCCS-1)] ))
  

    def featxtr(self, soundfiles, labels):
        
        ''' extract features from a list of sound files.
        
        For each sound file, extract features according to the method in
        `self.analyser`.
        
        Args:
            soundfiles list(str):   WAVE file paths
            labels [2d bool array]: ground truth
            
        Return:
            X (2d-array):           extracted features
        
        '''

        if labels == None:
            labels = [None for _ in range(len(soundfiles))]
            
        new_labels = []
        res = []
        
        soundfile_counter = 0
        total_counter     = 0
        chunk2soundfile   = {}
        
        if settings.MULTICORE:
            print ("Starting pool of %d processes" % (cpu_count()))
            pool = Pool(processes=cpu_count())

        for soundfile, lab in zip(soundfiles, labels):
        
#            print( "[%d.00/%d] analysing %s" % (soundfile_counter+1, len(soundfiles), os.path.basename(soundfile) ), end="" )
#            sys.stdout.flush()
            
            (rate,signal_all) = wav.read(soundfile)
            
            ''' take one channel only, if more than one present '''
            if len(signal_all.shape) > 1:    
                signal_all = signal_all[:,0]
                
            secs = float(len(signal_all))/rate
    
            if self.n_segments:
                signals = np.array_split(signal_all, self.n_segments)
                
            elif self.sec_segments:
                signals = [signal_all[x:x+rate*self.sec_segments] for x in np.arange(0, len(signal_all), rate*self.sec_segments)]
                #signals = np.array_split(signal_all, np.arange(0, len(signal_all), rate*self.sec_segments))
             
                
            else:
                signals = np.array([signal_all])
                
            # print ("signals:", signals)
            # print ("secs:", secs)
            # print ("sec per signal:", ", ".join(map(str, [len(x) for x in signals])))
            
            chunk_counter = 0
            for signal in signals:
                
                if self.sec_segments and len(signal) < self.sec_segments * rate: 
                    continue
            
                ''' some stats '''
                
                if secs > self.max_sound_length:
                    self.max_sound_length = secs
                
                if secs < self.min_sound_length:
                    self.min_sound_length = secs
                    
                self.total_sound_length += secs
                
                chunk_name = "%s%s" % (misc.mybasename(soundfile), ("_c%03d" % chunk_counter if self.n_segments or self.sec_segments else ""))
                
                    
                picklename = misc.make_output_filename(chunk_name, settings.analyser,
                                                       settings.modelname, "pickle", removeext=False)
                
                if not os.path.exists(picklename) or settings.FORCE_FEATXTR:
                    
                    is_analysing = True
                    
                    if settings.MULTICORE:
                        res.append( pool.apply_async(exec_featextr, 
                                        [soundfile, signal, rate, self.analyser, picklename,
                                         soundfile_counter, chunk_counter, len(soundfiles), 
                                         self.highpass_cutoff, self.normalise]) )
                    else:
                        res.append( exec_featextr(soundfile, signal, rate, self.analyser, picklename,
                                         soundfile_counter, chunk_counter, len(soundfiles), 
                                         self.highpass_cutoff, self.normalise) )
                      
                else:
                    print( "\r[%d.%02d/%d] unpickling %s" % (soundfile_counter+1, chunk_counter, len(soundfiles), os.path.basename(picklename) ), end="" )
                    is_analysing = False
                    res.append(misc.load_from_pickle(picklename))
             
                chunk2soundfile[total_counter] = soundfile
                chunk_counter += 1
                total_counter += 1
                
                new_labels.append(lab)
                ### end signals loop
                
            print( "\r[%d.%02d/%d] %s %s         " % (soundfile_counter+1, chunk_counter, len(soundfiles), "analysed" if is_analysing else "unpickled", os.path.basename(soundfile if is_analysing else picklename) ), end="" )
            sys.stdout.flush()
            
            soundfile_counter += 1
            ### end soundfiles loop
            
            
        if settings.MULTICORE:
            pool.close()
            pool.join()
        
            
        X = [] ## (n_samples x n_features)
        for x in res:
            
            if isinstance(x, ApplyResult):
                x = x.get()

            if x == None: continue
            
            X.append(x)

        
        X = np.vstack(X)
        new_labels = np.vstack(new_labels)

        assert X.shape[0] == new_labels.shape[0]
        print()
        
        self.chunk2soundfile = chunk2soundfile
        
        return X, new_labels
    
        
    def export_for_azure(self, X, labels, n_soundfiles):
        
        temp_labels = labels.reshape(len(labels),1)
        Y = np.hstack( (temp_labels, X.astype('S8')) )
        
        # sanity check     
        assert X.shape[0] == Y.shape[0] == temp_labels.shape[0] == n_soundfiles
        #assert X.shape[1] * settings.NFEATURES +1 == Y.shape[1]
        
        self.savedata("training" , Y[::2] ) # every even row in X
        self.savedata("comparing", Y[1::2]) # every odd row in X
        
        return Y
        
        
    def preprocess(self, data_training, data_testing):
        
        print ("training shape: ", data_training.shape)
        if settings.whiten_feature_matrix:
            data_training = whiten(data_training)
            data_testing  = whiten(data_testing)
    
        if settings.FEATURES_PLOT:
            feature_plot(data_training, settings.modelname, settings.MPL_FORMAT)
        
        return data_training, data_testing

    def classify(self, data_training, labels_training, data_testing, labels_testing):
        
        if not settings.MULTILABEL:
            labels_training = np.ravel(labels_training)
            labels_testing  = np.ravel(labels_testing)
        
                
        self.labels_testing = labels_testing
        self.labels_training = labels_training
        self.data_training = data_training
        self.data_testing = data_testing
        
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
            self.predict_proba = self.predict_proba.T
        
        if not self.missingTestLabels:
            

            if settings.MULTILABEL:
                if settings.CLASSIF_PLOT: classif_plot(labels_testing, self.prediction)
                
                auc = roc_auc_score(labels_testing, self.predict_proba)
                
                tp = np.sum((labels_testing + self.prediction) == 2)
                fp = np.sum((labels_testing - self.prediction) == -1)
                fn = np.sum((labels_testing - self.prediction) == 1)
                print (colored("ROC AUC: %.3f" % auc, "green"))
                print ("TP:", tp)
                print ("FP:", fp)
                print ("FN:", fn)
                self.results["auc"] = auc
                self.results["tps"] = tp
                self.results["fps"] = fp
                self.results["fns"] = fn
                
            else: # single label
                
                print ("classes:")
                print (self.classes)
                print ("test labels:")
                print (labels_testing)
                print ("prediction:")
                print (self.prediction)
                confusion = ConfusionMatrix(self.classes)
                res_colored = ""
                for true, pred in zip(labels_testing, self.prediction):
                #for true, pred in np.hstack((labels_testing, self.prediction)):
                    print (colored("true %s, predicted %s" % (true, pred), "green" if true == pred else "red"))
                    res_colored += "<span style='color: %s;'>true %s, predicted %s</span><br>" % ("green" if true == pred else "red", true, pred)
                    confusion.add(str(true), str(pred))
                
                self.results["colored_res"] = res_colored
                if settings.CONFUSION_PLOT:
                    print (confusion)
                    confusion.plot(outputname=misc.make_output_filename("confusion", "", settings.modelname, settings.MPL_FORMAT))
                self.results["confusion_matrix"] = confusion.html()
                self.confusion = confusion
                
            
            res = np.ravel(labels_testing) == np.ravel(self.prediction)
            self.res = res
            
            f1  = f1_score(labels_testing, self.prediction)
            self.results["f1"]  = f1            
            print (colored("f1: %.3f" % f1, "green"))
            
            if settings.MULTILABEL:
                total = np.sum(self.test_labels)

                self.results["correct"] = tp            
                self.results["total"]   = total
                self.results["correct_percent"] = float(tp)/total*100
                
#                print ("correct %d/%d (%.3f%%)" % (self.results["correct"], self.results["total"], self.results["correct_percent"]))
#                print ("correct %d/%d (%.3f%%)" % (tp, int(np.sum(self.test_labels)), tp/np.sum(self.test_labels)*100))

            else:
                total = len(self.test_labels)
                self.results["correct"] = np.sum(res)
                self.results["total"]   = np.size(res)
                self.results["correct_percent"] = np.sum(res)/float(np.size(res))*100
                
                

            print ("correct %d/%d (%.3f%%)" % (self.results["correct"], self.results["total"], self.results["correct_percent"]))
                
            
            

            
    def run(self):
        '''
        Execute an example of the module usage.
        
        * extract features for all appropriate files in soundpath
        * classify locally
        * print if it was correctly classified
            
        '''
        
        print (colored ("\nstarting %s" % type(self).__name__, "red"))
        print ("classes are:", settings.CLASSES)
                    
        print (colored ("[%s] analysing training files..." % settings.analyser, "red"))
        self.total_sound_length, self.min_sound_length, self.max_sound_length = 0, np.inf, 0
        data_training, self.train_labels = self.featxtr(self.train_soundfiles, self.train_labels)
        self.results["min_sound_length_train"] = self.min_sound_length
        self.results["max_sound_length_train"] = self.max_sound_length
        self.results["avg_sound_length_train"] = self.total_sound_length / self.train_labels.shape[0]
        
        
        if settings.SPLIT_TRAINING_SET:
            
            test_size = 0.5
            
            print (colored("WARNING: Splitting training set for train/test data (%.2f)" % test_size, "red"))
            
            data_training, data_testing,\
            self.train_labels, self.test_labels = train_test_split(data_training, 
                                                                   self.train_labels,
                                                                   test_size=test_size, random_state=88)
                                                                   
            self.total_sound_length *= test_size
            self.results["n_train_files"] = len(self.train_labels)
            self.results["n_test_files"] = len(self.test_labels)
            
            #data_testing, self.test_labels = data_training, self.train_labels
            #print (colored ("[%s] skipping test files (splitting training set)" % settings.analyser, "red"))
        
            
        else:
            print (colored ("[%s] analysing testing files..." % settings.analyser, "red"))
            self.total_sound_length, self.min_sound_length, self.max_sound_length = 0, np.inf, 0
            data_testing, self.test_labels  = self.featxtr(self.test_soundfiles, self.test_labels)
        self.results["min_sound_length_test"] = self.min_sound_length
        self.results["max_sound_length_test"] = self.max_sound_length
        self.results["avg_sound_length_test"] = self.total_sound_length / self.test_labels.shape[0]
        
        print (colored ("[%s] preprocessing..." % settings.analyser, "red"))
        data_training, data_testing = self.preprocess(data_training, data_testing)
        
        self.data_training, self.data_testing = data_training, data_testing
        self.results.update({"nsamples"  : data_training.shape[0],
                             "nfeatures" : data_training.shape[1], })
        if not self.missingTestLabels:
            if settings.MULTILABEL:
                self.results["items_to_classify"] = np.sum(self.test_labels)
            else:
                self.results["items_to_classify"] = len(data_testing)                
            
            
        
        
        # CODE HAS CHANGED. EXPORT FOR AZURE NEEDS ADJUSTING AND TESTING  
        #print (colored ("exporting data...", "red"))
        #Y = self.export_for_azure(self.X, self.labels, len(self))
        
        print (colored ("[%s] classifying..." % settings.classifier, "red"))
        if settings.classifier == 'decisiontree' or settings.classifier == 'randomforest':
#            settings.MULTILABEL = True
#            labels_testing = label_binarize(self.train_labels, self.classes)
#            labels_training = label_binarize(self.test_labels, self.classes)
#            self.classify(data_training, labels_training, data_testing, labels_testing)
            
#             settings.MULTILABEL = False
             self.classify(data_training, self.train_labels, data_testing, self.test_labels)
            

            
        elif settings.classifier == "kmeans":
            self.cluster(self.data_testing)
        else:
            raise Exception("Classifier not implemented.")
        
        misc.dump_report(self.results, self.modelname)