import os
from scipy.cluster.vq import whiten
from classispecies import settings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize

from classispecies.utils.confusion import ConfusionMatrix

settings.modelname  = "nips4b"
settings.classifier = "randomforest"
settings.analyser   = "mel-filterbank"
settings.CONFUSION_PLOT = False

settings.SOUNDPATHS.update({
    'Boa' :
        { 'train' : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train',
          'test'  : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train',
        },
    })
settings.LABELS = {
    'Boa' :
        { 'train' : '/home/dz2v07/cicada-largefiles/NIPS4B/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/numero_file_train.csv',
          'test'  : '',
        },
    }
settings.CLASSES = range(87)
settings.NCLASSES = 87
settings.SPLIT_TRAINING_SET = True
settings.FORCE_FEATXTR = False

if settings.analyser == "mfcc":
    settings.NMFCCS = 17
    settings.NFEATURES = settings.NMFCCS 
    
elif settings.analyser == "mel-filterbank":
    settings.NFEATURES = 26*3
    
elif settings.analyser == "hertz-spectrum":
    pass

else:
    raise Exception("[%s] invalid analyser" % __file__)

#from classispecies import classispecies
#classispecies.run()

from classispecies.classispecies import Classispecies
from classispecies.utils import misc

class NipsModel(Classispecies):

    
    def classify(self, data_training, labels_training, data_testing, labels_testing):
        
        # self.classify2(data_training)
        # return
        
        labels = np.loadtxt(settings.LABELS[misc.gethostname()]['train'], delimiter=",")[:,1:-1]
        labels_training = labels[0::2]
        labels_testing  = labels[1::2]
        
        # labels_training = np.loadtxt(settings.LABELS[misc.gethostname()]['train'], delimiter=",")[:,1:-1]
        # labels_testing = np.loadtxt(settings.LABELS[misc.gethostname()]['train'], delimiter=",")[:,1:-1]
        
        self.set_classes(range(labels_training.shape[1])) # 0..87
        
        
        if self.classifier == "decisiontree":
            clf = DecisionTreeClassifier()
        elif self.classifier == "randomforest":
            clf = RandomForestClassifier()
        else:
            raise Exception("Unknown classifier")
        
        clf = clf.fit(data_training, labels_training)
        
        self.prediction = clf.predict(data_testing)#.reshape(len(testing),1)
        
        self.clf = clf
        self.data_testing = data_testing
        self.labels_testing = labels_testing
        
        
        if labels_testing != None:
            res = self.prediction == labels_testing
            self.res = res
        
            self.results["correct"] = np.sum(res)
            self.results["total"]   = np.size(res)
            self.results["correct_percent"] = np.sum(res)/float(np.size(res))*100
            print ("correct %d/%d (%.3f%%)" % (self.results["correct"], self.results["total"], self.results["correct_percent"]))
            
            if settings.CLASSIF_PLOT:
                    
                print ("plotting classification plot")
                fig = plt.figure(figsize=(15,7))
                ax = fig.add_subplot(131)
                ax.autoscale(tight=True)
                ax.pcolormesh(labels_testing, rasterized=True)
                ax.set_title("ground truth")
                
                ax = fig.add_subplot(132)
                ax.autoscale(tight=True)
                ax.pcolormesh(self.prediction, rasterized=True)
                ax.set_title("prediction")
                
                ax = fig.add_subplot(133)
                
                img = ax.pcolormesh(labels_testing - self.prediction, cmap=cm.get_cmap('PiYG',3), rasterized=True)
                cb = plt.colorbar(img, ax=ax)
                cb.set_ticks([-1,0,1], False)
                cb.set_ticklabels(["FP", "T", "FN"], False)
                cb.update_ticks()
                ax.autoscale(tight=True)
                
                print ("saving classification plot")
            
                # fig.savefig(misc.make_output_filename("classifplot", "classify", settings.modelname, settings.MPL_FORMAT), dpi=150)
                fig.savefig(misc.make_output_filename("classifplot", "classify", settings.modelname, "png"), dpi=150)
                
                fig.show()
            
            #print "ROC AUC:", roc_auc_score(labels_testing, data_testing)
            auc = roc_auc_score(labels_testing, self.prediction)
            tp = np.sum((labels_testing + self.prediction) == 2)
            fp = np.sum((labels_testing - self.prediction) == -1)
            fn = np.sum((labels_testing - self.prediction) == 1)
            print "ROC AUC:", auc
            print "TP:", tp
            print "FP:", fp
            print "FN:", fn
            self.results["auc"] = auc
            self.results["tps"] = tp
            self.results["fps"] = fp
            self.results["fns"] = fn
            
            
    def classify2(self, data_training):
        
        global probas_, scores, y, X_train, X_test, y_train, y_test, classifier
        X = data_training
        y = np.loadtxt(settings.LABELS[misc.gethostname()]['train'], delimiter=",")[:,1:-1]
        # Binarize the output
        y = label_binarize(y, classes=range(87))
        n_classes = y.shape[1]

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                            random_state=0)
        
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        probas_     = classifier.predict_proba(X_test)
        prediction  = classifier.predict(X_test)
        #y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        scores = cross_val_score(classifier, X_test, y_test)
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]))
        #for i in range(n_classes):
        #    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
        #                                   ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
        
        auc = roc_auc_score(y_test, X_test)
        print "ROC AUC:", auc
        self.results["auc"] = auc


        

    def dump_to_nips4b_submission(self):
        with open(misc.make_output_filename("nips4b_bird2013_sidelil_1", "submission", settings.modelname, "csv"), 'w') as f:
            f.write("ID,Probability\n")
            nrow=1
            for row in self.prediction:
                ncol=1
                for column in row:
                    f.write("nips4b_birds_testfile%04d.wav_classnumber_%d,%f\n" %(nrow, ncol, column))
                    ncol+=1
                nrow+=1
        


model = NipsModel(analyser=settings.analyser, classifier=settings.classifier)
model.run()
model.dump_to_nips4b_submission()