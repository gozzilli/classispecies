# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:44:16 2014

@author: Davide Zilli
"""

from __future__ import print_function

import sys, os
import traceback

import numpy as np
import scipy
import scipy.io.wavfile as wav

from scipy.cluster.vq import  whiten

from features import mfcc, logfbank, fbank
from stowell.oskmeans import OSKmeans, normalise_and_whiten

from multiprocessing.pool import ApplyResult
from multiprocessing import Pool, cpu_count


import settings
from utils import misc, signal as usignal
from utils.misc import rprint
from utils.plot import feature_plot

logger = misc.config_logging("classispecies")

def downscale_spectrum(feat, target):
    ''' feat must be the right way up already (i.e. for fbank and the lot, transpose it first) '''

    out = []
    no_y, no_x = feat.shape
    
    incr = no_x/float(target)
    
    for i in range(target):
        #print "A", feat[:,incr*i:incr*(i+1)]
        #print "B", mean(feat[:,incr*i:incr*(i+1)], axis=1)[np.newaxis]
        out.append( np.mean(feat[:,incr*i:incr*(i+1)], axis=1)[np.newaxis].T )

    return np.array(np.hstack(out))
        
    
    
""" OLD DOWNSCALE SPECTRUM  
def downscale_spectrum(feat, target):
    ''' feat must be the right way up already (i.e. for fbank and the lot, transpose it first) '''
    
    feats = []
        
    no_y, no_x = feat.shape
    x = range(no_x)
    bin_counter = 0
    
    for bin_ in feat:
    
        f = interp1d(x, bin_)
        xnew = np.arange(target)
        fx = f(xnew)
        feats.append(fx)
        bin_counter += 1
        
    return np.array(feats)
"""

def half2Darray(arr):
    return arr[:,:arr.shape[1]/2]


class FeatureSet(object):
    
    data        = None
    labels      = None
    max_length  = None
    min_length  = None
    avg_length  = None
    tot_length  = None
    db          = None
    
    def extract(self, soundfiles, labels, analyser=None, highpass_cutoff=None, 
                normalise=None, n_segments=None, sec_segments=None):
    
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
            labels = np.empty( (len(soundfiles), 1) )
            
        new_labels = []
        res = []
        
        soundfile_counter = 0
        total_counter     = 0
        chunk2soundfile   = []
        db = []
        
        allpicklename = misc.make_output_filename("all", settings.analyser+str(settings.sec_segments or ""),
                                                       settings.modelname, "pickle", removeext=False)
                                                       
        if not os.path.exists(allpicklename) or settings.FORCE_FEATXTRALL:
            
            if settings.MULTICORE:
                logger.info ("Starting pool of %d processes" % (cpu_count()))
                pool = Pool(processes=cpu_count())
                
            picklenames = []
    
            for soundfile, lab in zip(soundfiles, labels):
            
    #            logger.info( "[%d.00/%d] analysing %s" % (soundfile_counter+1, len(soundfiles), os.path.basename(soundfile) ), end="" )
    #            sys.stdout.flush()
                
                try:
                    (rate,signal_all) = wav.read(soundfile)
                except:
                    print ("\n", soundfile, "\n")
                    raise
                
                ''' take one channel only, if more than one present '''
                if len(signal_all.shape) > 1:    
                    signal_all = signal_all[:,0]
                    
                #secs = float(len(signal_all))/rate
        
                if n_segments:
                    signals = np.array_split(signal_all, n_segments)
                    
                elif sec_segments:
                    signals = [signal_all[x:x+rate*sec_segments] for x in np.arange(0, len(signal_all), rate*sec_segments)]
                    #signals = np.array_split(signal_all, np.arange(0, len(signal_all), rate*self.sec_segments))
                 
                    
                else:
                    signals = np.array([signal_all])
                    
                # logger.info ("signals:", signals)
                # logger.info ("secs:", secs)
                # logger.info ("sec per signal:", ", ".join(map(str, [len(x) for x in signals])))
                
                
                chunk_counter = 0
                for signal in signals:
                
                    if sec_segments and len(signal) < sec_segments * rate: 
                        continue
                
                    chunk2soundfile.append(soundfile_counter)
                    db.append( (soundfile, chunk_counter, len(signal)/float(rate), lab) )
    #                if total_counter % 10000 == 0:
    #                    print ("\n{:,} kB\n".format(sum([sys.getsizeof(x) for x in [
    #                        self.db, db, self.data, labels, self.max_length, self.min_length,
    #                        self.avg_length, self.tot_length, picklenames, total_counter, 
    #                        signals, new_labels]])/1000.))
    #                    print ("1  {:,}".format(sys.getsizeof(db)/1000))
    #                    print ("2  {:,}".format(sys.getsizeof(self.db)/1000))
    #                    print ("3  {:,}".format(sys.getsizeof(self.data)/1000))
    #                    print ("4  {:,}".format(sys.getsizeof(labels)/1000))
    #                    print ("5  {:,}".format(sys.getsizeof(self.max_length)/1000))
    #                    print ("6  {:,}".format(sys.getsizeof(self.min_length)/1000))
    #                    print ("7  {:,}".format(sys.getsizeof(self.avg_length)/1000))
    #                    print ("8  {:,}".format(sys.getsizeof(self.tot_length)/1000))
    #                    print ("9  {:,}".format(sys.getsizeof(res)/1000))
    #                    print ("10 {:,}".format(sys.getsizeof(picklenames)/1000))
    #                    print ("11 {:,}".format(sys.getsizeof(new_labels)/1000))
    #                    print ("12 {:,}".format(sys.getsizeof(total_counter)/1000))
    #                    print ("13 {:,}".format(sys.getsizeof(signals)/1000))
    #                    print ()
                            
                    
                    chunk_name = "%s%s" % (misc.mybasename(soundfile), ("_c%03d" % chunk_counter if n_segments or sec_segments else ""))
                    
                        
                    picklename = misc.make_output_filename(chunk_name, settings.analyser+str(settings.sec_segments or ""),
                                                           settings.modelname, "pickle", removeext=False)
                    picklenames.append(picklename)
                    
                    if not os.path.exists(picklename) or settings.FORCE_FEATXTR:
                        
                        is_analysing = True
                        
                        if settings.MULTICORE:
                            #res.append( pool.apply_async(exec_featextr, 
                            pool.apply_async(exec_featextr,
                                            [soundfile, signal, rate, analyser, picklename,
                                             soundfile_counter, chunk_counter, len(soundfiles), 
                                             highpass_cutoff, normalise]) 
                            #)
                        else:
                            #res.append( exec_featextr(soundfile, signal, rate, analyser, picklename,
                            exec_featextr(soundfile, signal, rate, analyser, picklename,
                                             soundfile_counter, chunk_counter, len(soundfiles), 
                                             highpass_cutoff, normalise) 
                            #)
                          
                    else:
                        if soundfile_counter % 30 == 0:
                            rprint( "[%d.%02d/%d] unpickling %s" % (soundfile_counter+1, chunk_counter, len(soundfiles), os.path.basename(picklename) ))
                        is_analysing = False
                        #res.append(misc.load_from_pickle(picklename))
                        
                    #print ("{:,}".format(sys.getsizeof(res)/1000))
                 
                    chunk_counter += 1
                    total_counter += 1
                    
                    new_labels.append(lab)
                    ### end signals loop
                    
                #rprint( "[%d.%02d/%d] %s %s         " % (soundfile_counter+1, chunk_counter, len(soundfiles), "analysed" if is_analysing else "unpickled", os.path.basename(soundfile if is_analysing else picklename) ) )
                #sys.stdout.flush()
                
                # gc.collect()
                
                soundfile_counter += 1
                ### end soundfiles loop
                
            print() 
        
        
            if settings.MULTICORE:
                pool.close()
                pool.join()
                
            print ("unpickling %d files" % len(picklenames))
            for picklename in picklenames:
                res.append(misc.load_from_pickle(picklename))
            
                
            X = [] ## (n_samples x n_features)
            res_counter = 0
            for x in res:
                
                res_counter += 1
                if isinstance(x, ApplyResult):
                    print ("%d ApplyResult" % res_counter)
                    x = x.get()
    
                if x == None: 
                    print ("%d continuing..." % res_counter)
                    continue
                
                X.append(x)
    
            
            X = np.vstack(X)
            new_labels = np.vstack(new_labels)
            
            misc.dump_to_pickle( (X, new_labels, db), allpicklename)
        
        else:
            X, new_labels, db = misc.load_from_pickle(allpicklename)

        print("\nX.shape[0]: %d, new_labels.shape[0]: %d" % (X.shape[0], new_labels.shape[0]))
        print()
        assert X.shape[0] == new_labels.shape[0]
        logger.info("")
        
        db = np.array(db, dtype=[("soundfile"        , 'S200'), 
                                 ("chunk_counter"    , int),
                                 ("length"           , int),
                                 ("truth"            , str(labels.shape[-1])+"S")]).view(np.recarray)

        
        self.data = X
        self.labels = db.truth
        self.db = db
        self.set_stats()
        
        return X, db
        
    def set_stats(self):
        self.max_length = np.max(self.db.length)
        self.min_length = np.min(self.db.length)
        self.avg_length = np.mean(self.db.length)

        
        
    def split(self, soundfiles, cut=0.5):
        
        assert cut*len(self) # must not be 0 or None
        print (self.labels)
        indices = np.random.permutation(len(self))
        idx1, idx2 = indices[:cut*len(self)], indices[cut*len(self):]
        
        data1, data2 = self.data  [idx1,:], self.data  [idx2,:]
        if settings.MULTILABEL:
            lab1,  lab2  = self.labels[idx1,:], self.labels[idx2,:]    
        else:
            lab1, lab2 = self.labels[idx1], self.labels[idx2]

        db1, db2 = self.db[idx1], self.db[idx2]
        
        train_soundfiles = []
        test_soundfiles  = []
        
        print ("DBs", len(db1.soundfile), len(db2.soundfile))
        
        missing = []
        
        for f in soundfiles:
            if f in db1.soundfile:
                train_soundfiles.append(f)
            if f in db2.soundfile:
                test_soundfiles.append(f)
            if f not in db1.soundfile and f not in db2.soundfile:
                print (f, "missing")
                missing.append(f)
                #raise Exception("Soundfile %s doesn't seem to belong to neither train nor test set" % f)
        
        print ("missing (%d) %s" % (len(missing), missing))
        
        print ("train soundfiles length:", len(train_soundfiles))
        print ("test  soundfiles length:", len(test_soundfiles))
        
        fs1 = FeatureSet()
        fs1.db = db1
        fs1.data = data1
        fs1.labels = lab1
        fs1.set_stats()
        
        fs2 = FeatureSet()
        fs2.db = db2
        fs2.data = data2
        fs2.labels = lab2
        fs2.set_stats()
        
        return fs1, fs2, [],[]#train_soundfiles, test_soundfiles
        
    def preprocess(self):
        
        if settings.whiten_feature_matrix:
            self.data = whiten(self.data)
        
    
    def __len__(self):
        return len(self.data)
    
    

def aggregate_feature(feat, normalise=False):
    ''' Take any combination of mean, std and max of the given features
    
    according to the parameters in settings. 
    
    Args
        feat             -- the 2D array of features
        normalise (bool) -- normalise or not?
        
    Return
        a horizontally stacked vector (1-row array) such as:
        [mean1 mean2 mean3 var1 var2 var3 max1 max2 max3]
        
    '''
    
    out_feat = []
    func = {np.nanmean:settings.extract_mean,
            np.nanstd:settings.extract_std,
            np.nanmax:settings.extract_max}
    func = [x for x in func.keys() if func[x]]
    
    if normalise:
        feat = usignal.rms_normalise(feat)

    for f in func:
        
        r = f(feat, axis=0)
        if settings.whiten_feature:
            r = whiten(r)
        out_feat.append(r)
        
    feats = np.hstack(out_feat)
    if settings.whiten_features:
        feats = whiten(feats)
    return feats
        
def slice_spectrum(feat, delta):
    '''
    Stack the spectrum in slices, such that `delta` consecutive slices 
    appear as one row. 
    
    If `delta` is 0, return the input spectrum.
    '''
    
    if delta > 0:
            
        slices = np.empty((np.floor(feat.shape[0] / delta), feat.shape[1]*delta))
        if np.floor(feat.shape[0] / delta) == 0:
            return None
            
        for i in range(0, slices.shape[0]):
            
            slices[i] = np.ravel(feat[i*delta:i*delta+delta, :])
    
    else:
        slices = feat
    return slices
        

def extract_mfcc(signal, rate, normalise):
    
    mfcc_feat = np.clip(np.nan_to_num(mfcc(signal, rate, numcep=settings.NMFCCS)), 
                            a_min=-100, a_max=100)
    return aggregate_feature(mfcc_feat, normalise)
    
def extract_mel(signal, rate, normalise):

#    global mel_feat, slices, mf, mfb, signal2
#    signal2 = signal
    
#    mf = fbank(signal, rate, lowfreq=500, 
#                                              winlen=0.0232, winstep=0.0232,
#                                           nfilt=settings.N_MEL_FILTERS)
#    mel_feat = np.nan_to_num(mf[0])
    mfb = logfbank(signal, rate, lowfreq=500, 
#                                              winlen=0.0232, winstep=0.0232,
                                           nfilt=26)#settings.N_MEL_FILTERS)
    mel_feat = np.clip(np.nan_to_num(mfb), 
                       a_min=-100, a_max=100)            
    
    slices = slice_spectrum(mel_feat, settings.delta)
    
#    assert np.max(slices) <= 100
#    assert np.min(slices) >= -100

    feat = aggregate_feature(slices, normalise)
    return feat
    
def extract_melfft(signal, rate, normalise):    
    
    NFFT = 2**14
    NFILT = 40
    
#    mel_feat = np.clip(np.nan_to_num(logfbank(signal, rate, lowfreq=500, 
#                                              #winlen=0.0232, winstep=0.0232,
#                                              #nfilt=settings.N_MEL_FILTERS)), 
#                                             nfilt=26)),
#                       a_min=-100, a_max=100)
    
    mel_feat = fbank(signal, nfilt=NFILT, winstep=0.0232, winlen=0.0232)[0]
    melfft = np.abs(scipy.fft(mel_feat.T, n=NFFT))[:,0:NFFT/2]
    
    try:
        assert melfft.shape == (NFILT, NFFT/2)
    except AssertionError:
        print ("MELFFT SHAPE:", melfft.shape)
        raise
        
    return np.hstack(10*np.log10(downscale_spectrum(melfft, 10)))
    
    
    d, M = mel_feat.shape
    x = np.arange(d)

#    feats = []
#    bin_counter = 0
#    for bin_ in mel_feat.T:
#        fft_ = np.fft.fft(bin_)            
#        f = interp1d(x, fft_)
#        xnew = np.arange(10)
#        fx = f(xnew)
#        feats.append(fx)
#        bin_counter += 1

    feats = []
    for bin_ in mel_feat.T:
        fft_ = np.fft.fft(bin_, n=20)
        feats.append(fft_[0:10])
    
    return np.hstack( feats )  

def extract_hertzfft(signal, rate, normalise):
    
    NFFT = 2**14
    
    frame_size = 256
    hop        = 256

    hertz_feat = half2Darray(np.abs(usignal.stft2(signal, rate, frame_size, hop))).T
    hertz_fft  = half2Darray(np.abs(scipy.fft(hertz_feat, n=NFFT)))
    
    return np.hstack(10*np.log10(downscale_spectrum(hertz_fft, 10)))
    
def extract_hertz(signal, rate, normalise):
    
    signal__ = signal
#        hertz_feat = usignal.stft(signal, rate, 0.1, 0.01)        
#        hertz_feat = 10*np.log10(hertz_feat.T[:len(hertz_feat.T)/2])
#        hertz_feat = np.clip(np.nan_to_num(hertz_feat), a_min=-100, a_max=100)
 
    frame_size = 0.0018 # for 1sec of signal, 0.0005 = 22 bins, then divided by two 11 (22)
    hop = 0.003 # also noverlap in specgram, roughly every 128 samples

    hertz_feat = 10*np.log10(usignal.stft(signal, rate, frame_size, hop))
    hertz_feat = hertz_feat[:, 0:hertz_feat.shape[1]/2]
    
    slices = slice_spectrum(hertz_feat, settings.delta)
    
    feat = aggregate_feature(slices, normalise)
    return feat
    

def extract_oskmeans(signal, rate, normalise):
        
    mel_feat = np.clip(np.nan_to_num(logfbank(signal, rate, 
                                              lowfreq=500, 
                                              nfilt=settings.N_MEL_FILTERS)), 
                       a_min=-100, a_max=100)
    
    
    #info, nw = normalise_and_whiten(mel_feat)
    nw = mel_feat
    
    k = 500    
    d = nw.shape[1]
    oskm = OSKmeans(k, d)
    
    for datum in nw:
        oskm.update(datum)

    feat = np.hstack( (np.mean(oskm.centroids, axis=1), 
                       np.std (oskm.centroids, axis=1)) )
    
    assert feat.shape == (1000,)
    print()
    return feat
        

def exec_featextr(soundfile, signal, rate, analyser, picklename,
                  soundfile_counter, chunk_counter, n_soundfiles, 
                  highpass_cutoff, normalise):
    try:
        rprint("[%d.%02d/%d] [pid:%d] analysing %s" % (soundfile_counter+1, chunk_counter, 
                                  n_soundfiles, os.getpid(), os.path.basename(soundfile)) )
        
    #    if highpass_cutoff > 0:
    #        signal = usignal.highpass_filter(signal[:], cutoff=highpass_cutoff)
                
        if   analyser == "mfcc"           : feat = extract_mfcc(signal, rate, normalise)
        elif analyser == "mel-filterbank" : feat = extract_mel(signal, rate, normalise)
        elif analyser == "melfft"         : feat = extract_melfft(signal, rate, normalise)
        elif analyser == "hertz-spectrum" : feat = extract_hertz(signal, rate, normalise)
        elif analyser == "hertzfft"       : feat = extract_hertzfft(signal, rate, normalise)
        elif analyser == "oskmeans"       : feat = extract_oskmeans(signal, rate, normalise)
        else:
            raise ValueError("Feature extraction method '%s' not known." % analyser)
        
        misc.dump_to_pickle(feat, picklename)
        
        return feat
    except:
        
        traceback.print_exc()
