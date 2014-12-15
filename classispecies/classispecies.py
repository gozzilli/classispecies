# -*- coding: utf-8 -*-

from __future__ import print_function
import os
from termcolor import colored
from abc import ABCMeta
from collections import OrderedDict
from pprint import pprint


import settings


import numpy as np
import scipy.io.wavfile as wav
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult

from scipy.cluster.vq import whiten
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split

from featextr import exec_featextr, FeatureSet
from utils.confusion import ConfusionMatrix
from utils.plot import classif_plot, feature_plot
from utils import misc
from utils.misc import rprint

mel_feat = None
signal__ = None

max__ = 0
max__file = ""

IMPORTANT = "blue"
WARNING   = "red"
RESULT    = "green"
POSITIVE  = "green"
NEGATIVE  = "red"

logger = misc.config_logging("classispecies")


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
                        "n_test_files"  : len(self.test_soundfiles) if self.test_soundfiles != None else 0,
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
            for root, subFolders, files in os.walk(misc.get_soundpath(set_=set_)):
                logger.info ("selecting files in %s" % root)
                
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
        
        return np.array(train_soundfiles), np.array(test_soundfiles)
    
    
    def savedata(self, filename, obj):
        ''' export feature data to file so that they can be processed by Azure ML '''
        
        np.savetxt(misc.make_output_filename(filename, "", settings.modelname, "csv"), obj, delimiter=",",
                   fmt="%s", comments = '', header= ",".join(["label"] +
                                    #["mean%d" % x for x in range(settings.NMFCCS-1)] +
                                    #["std%d"  % x for x in range(settings.NMFCCS-1)] +
                                    ["max%d"  % x for x in range(settings.NMFCCS-1)] ))
  
        
    def featxtr3(self):
        
        if settings.SPLIT_TRAINING_SET:

            logger.warn (colored("Splitting training set for train/test data (%.2f)" % settings.test_size, WARNING))
            
            cut = settings.test_size
            nsamples = len(self.train_soundfiles)
            indices = np.random.permutation(nsamples)
            idx_tr, idx_te = indices[:cut*nsamples], indices[cut*nsamples:]
            
            self.test_soundfiles  = self.train_soundfiles[idx_te]             
            self.train_soundfiles = self.train_soundfiles[idx_tr]
            
            self.test_labels  = self.train_labels[idx_te,:]
            self.train_labels = self.train_labels[idx_tr,:]
            
            self.unchunked_test_labels = self.test_labels.copy()
            

        
        ### TRAINING SET ######################################################
            
        logger.info (colored ("[%s] analysing training files..." % settings.analyser, IMPORTANT))
        
        train_fs = FeatureSet()
        train_fs.extract(self.train_soundfiles, self.train_labels, self.analyser, 
                         self.highpass_cutoff, self.normalise, 
                         self.n_segments, self.sec_segments)
        

        ### TE$T SET ##########################################################
        
        logger.info (colored ("[%s] analysing testing files..." % settings.analyser, IMPORTANT))

        test_fs = FeatureSet()
        test_fs.extract(self.test_soundfiles, self.test_labels, self.analyser, 
                        self.highpass_cutoff, self.normalise, 
                        self.n_segments, self.sec_segments)
                        

        self.results.update({
            "min_sound_length_train" : train_fs.min_length,
            "max_sound_length_train" : train_fs.max_length,
            "avg_sound_length_train" : train_fs.avg_length,
            "n_train_files"          : len(train_fs),
            "min_sound_length_test"  : test_fs.min_length,
            "max_sound_length_test"  : test_fs.max_length,
            "avg_sound_length_test"  : test_fs.avg_length,
            "n_test_files"           : len(test_fs),
            })
            
        pprint (self.results)
        print (len(train_fs.db.truth), len(test_fs.db.truth))
        
        
        return train_fs, test_fs

    def featxtr2(self):
        
        ### TRAINING SET ######################################################
            
        logger.info (colored ("[%s] analysing training files..." % settings.analyser, IMPORTANT))
        self.total_sound_length, self.min_sound_length, self.max_sound_length = 0, np.inf, 0
        
        data_training, self.train_labels, train_c2s = self.featxtr(self.train_soundfiles, self.train_labels)
        
        self.results["min_sound_length_train"] = self.min_sound_length
        self.results["max_sound_length_train"] = self.max_sound_length
        self.results["avg_sound_length_train"] = self.total_sound_length / self.train_labels.shape[0]
        

        ### TEST SET ##########################################################        
        if settings.SPLIT_TRAINING_SET:
            
            logger.warn (colored("Splitting training set for train/test data (%.2f)" % settings.test_size, WARNING))
            
            data_training, data_testing,\
            self.train_labels, self.test_labels = train_test_split(data_training, 
                                                                   self.train_labels,
                                                                   test_size=settings.test_size, random_state=88)
                                                                   
            self.total_sound_length *= settings.test_size
            self.results["n_train_files"] = len(self.train_labels)
            self.results["n_test_files"] = len(self.test_labels)
            
        else:

            logger.info (colored ("[%s] analysing testing files..." % settings.analyser, IMPORTANT))
            self.total_sound_length, self.min_sound_length, self.max_sound_length = 0, np.inf, 0
            data_testing, self.test_labels, self.test_c2s  = self.featxtr(self.test_soundfiles, self.test_labels)

        self.results["min_sound_length_test"] = self.min_sound_length
        self.results["max_sound_length_test"] = self.max_sound_length
        self.results["avg_sound_length_test"] = self.total_sound_length / self.test_labels.shape[0]
        
        return data_training, data_testing, self.train_labels, self.test_labels
        
    def featxtr4(self):            

        
        ### TRAINING SET ######################################################
            
        logger.info (colored ("[%s] analysing training files..." % settings.analyser, IMPORTANT))
        
        train_fs = FeatureSet()
        train_fs.extract(self.train_soundfiles, self.train_labels, self.analyser, 
                         self.highpass_cutoff, self.normalise, 
                         self.n_segments, self.sec_segments)

        if settings.SPLIT_TRAINING_SET:

            logger.warn (colored("Splitting training set for train/test data (%.2f)" % settings.test_size, WARNING))

            cut = settings.test_size
            print ("\n\n\nSPLITTING\n\n")
            train_fs, test_fs, self.train_soundfiles, self.test_soundfiles = train_fs.split(self.train_soundfiles, cut)
            
            u = OrderedDict()
            for soundfile, _, _, lab in test_fs.db:
                u[soundfile] = lab
            self.unchunked_test_labels = np.array(u.values())
            print ("len: utl:", self.unchunked_test_labels.shape)
        else:
            logger.info (colored ("[%s] analysing testing files..." % settings.analyser, IMPORTANT))
            test_fs = FeatureSet()
            test_fs.extract(self.test_soundfiles, self.test_labels, self.analyser, 
                            self.highpass_cutoff, self.normalise, 
                            self.n_segments, self.sec_segments)
            
        

        ### TE$T SET ##########################################################
        
        

        
        self.results.update({
            "min_sound_length_train" : train_fs.min_length,
            "max_sound_length_train" : train_fs.max_length,
            "avg_sound_length_train" : train_fs.avg_length,
            "n_train_files"          : len(train_fs),
            "min_sound_length_test"  : test_fs.min_length,
            "max_sound_length_test"  : test_fs.max_length,
            "avg_sound_length_test"  : test_fs.avg_length,
            "n_test_files"           : len(test_fs),
            })
            
        pprint (self.results)
        print (len(train_fs.db.truth), len(test_fs.db.truth))

        return train_fs, test_fs
        
        
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
        chunk2soundfile   = []
        
        if settings.MULTICORE:
            logger.info ("Starting pool of %d processes" % (cpu_count()))
            pool = Pool(processes=cpu_count())

        for soundfile, lab in zip(soundfiles, labels):
        
#            logger.info( "[%d.00/%d] analysing %s" % (soundfile_counter+1, len(soundfiles), os.path.basename(soundfile) ), end="" )
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
                
            # logger.info ("signals:", signals)
            # logger.info ("secs:", secs)
            # logger.info ("sec per signal:", ", ".join(map(str, [len(x) for x in signals])))
            
            chunk_counter = 0
            for signal in signals:
            
                
                if self.sec_segments and len(signal) < self.sec_segments * rate: 
                    continue
                
                chunk2soundfile.append(soundfile_counter)
            
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
                    rprint( "[%d.%02d/%d] unpickling %s" % (soundfile_counter+1, chunk_counter, len(soundfiles), os.path.basename(picklename) ))
                    is_analysing = False
                    res.append(misc.load_from_pickle(picklename))
             
                chunk_counter += 1
                total_counter += 1
                
                new_labels.append(lab)
                ### end signals loop
                
            rprint("[%d.%02d/%d] %s %s         " % (soundfile_counter+1, chunk_counter, len(soundfiles), "analysed" if is_analysing else "unpickled", os.path.basename(soundfile if is_analysing else picklename)[-30:] ) )
            
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
        logger.info("")
        
        assert len(chunk2soundfile) == total_counter
        
        return X, new_labels, chunk2soundfile
    
        
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
        
        logger.info ("training shape: %s" % str(data_training.shape))
        logger.info ("test shape    : %s" % str(data_testing.shape))
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
            
        c = OrderedDict()                
        for i in range(len(self.test_fs)):

            j = self.test_fs.db.soundfile[i]
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
                
                

            logger.info ("correct %d/%d (%.3f%%)" % (self.results["correct"], self.results["total"], self.results["correct_percent"]))
                
            ### Recombine the chunks
#            c = np.ones ( (len(self.test_soundfiles), labels_testing.shape[1]) ) 
#            c = OrderedDict()                
#            self.c = c
#            #print ("Shape of c:", c.shape)
#            for i in range(len(self.test_fs)):
#                j = self.test_fs.db.soundfile_counter[i]
#                #print (i,j, self.test_fs.db.soundfile[i][-10:])
#                if j in c:
#                    c[j] *= self.predict_proba[i]
#                else:
#                    c[j] = self.predict_proba[i]
#            self.c = np.array(c.values())
                
            
            logger.info (colored("ROC AUC merged: %.3f" % roc_auc_score(self.unchunked_test_labels, self.res_), RESULT))
            

            
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

        if settings.SPLIT_METHOD == "split-after":
            print ("split after (feaxtr4)")
            #data_training, data_testing, self.train_labels, self.test_labels = self.featxtr2()
            train_fs, test_fs = self.featxtr4()
            data_training, data_testing, self.train_labels, self.test_labels = \
                train_fs.data, test_fs.data, train_fs.db.truth, test_fs.db.truth
            self.test_fs = test_fs


        elif settings.SPLIT_METHOD in ["split-before", None]:        
            print ("split before (feaxtr3)")
            train_fs, test_fs = self.featxtr3()
            train_data, train_labs = train_fs.data, train_fs.db.truth
            test_data,  test_labs  = test_fs.data,  test_fs.db.truth
            
            data_testing = test_data
            data_training = train_data
            self.train_labels = train_labs
            self.test_labels = test_labs
            self.test_fs = test_fs
        else:
            raise ValueError("Cannot identify method %s for splitting training data" )

        ### PRE-PROCESSING ####################################################
        
        logger.info (colored ("[%s] preprocessing..." % settings.analyser, IMPORTANT))
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
        #logger.info (colored ("exporting data...", "red"))
        #Y = self.export_for_azure(self.X, self.labels, len(self))

        ### CLASSIFY ##########################################################
        
        logger.info (colored ("[%s] classifying..." % settings.classifier, IMPORTANT))
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