# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:44:16 2014

@author: Davide Zilli
"""

from __future__ import print_function

import sys, os
import numpy as np

import scipy.io.wavfile as wav

from scipy.interpolate import interp1d
from scipy.cluster.vq import  whiten

from features import mfcc, logfbank, fbank
from stowell.oskmeans import OSKmeans, normalise_and_whiten

from multiprocessing.pool import ApplyResult
from multiprocessing import Pool, cpu_count


import settings
from utils import misc, signal as usignal
from utils.misc import rprint

logger = misc.config_logging("classispecies")


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
                
                chunk_name = "%s%s" % (misc.mybasename(soundfile), ("_c%03d" % chunk_counter if n_segments or sec_segments else ""))
                
                    
                picklename = misc.make_output_filename(chunk_name, settings.analyser,
                                                       settings.modelname, "pickle", removeext=False)
                
                if not os.path.exists(picklename) or settings.FORCE_FEATXTR:
                    
                    is_analysing = True
                    
                    if settings.MULTICORE:
                        res.append( pool.apply_async(exec_featextr, 
                                        [soundfile, signal, rate, analyser, picklename,
                                         soundfile_counter, chunk_counter, len(soundfiles), 
                                         highpass_cutoff, normalise]) )
                    else:
                        res.append( exec_featextr(soundfile, signal, rate, analyser, picklename,
                                         soundfile_counter, chunk_counter, len(soundfiles), 
                                         highpass_cutoff, normalise) )
                      
                else:
                    rprint( "[%d.%02d/%d] unpickling %s" % (soundfile_counter+1, chunk_counter, len(soundfiles), os.path.basename(picklename) ))
                    is_analysing = False
                    res.append(misc.load_from_pickle(picklename))
             
                chunk_counter += 1
                total_counter += 1
                
                new_labels.append(lab)
                ### end signals loop
                
            rprint( "[%d.%02d/%d] %s %s         " % (soundfile_counter+1, chunk_counter, len(soundfiles), "analysed" if is_analysing else "unpickled", os.path.basename(soundfile if is_analysing else picklename) ) )
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
        logger.info("")
        
        global db
        db = np.array(db, dtype=[("soundfile"        , 'S200'), 
                                 ("chunk_counter"    , int),
                                 ("length"           , int),
                                 ("truth"            , str(labels.shape[1])+"int16")]).view(np.recarray)

        
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
        
        indices = np.random.permutation(len(self))
        idx1, idx2 = indices[:cut*len(self)], indices[cut*len(self):]
        
        data1, data2 = self.data  [idx1,:], self.data  [idx2,:]
        lab1,  lab2  = self.labels[idx1,:], self.labels[idx2,:]    

        print ("INDICES", len(idx1), len(idx2), idx1, idx2)
        
        db1, db2 = self.db[idx1], self.db[idx2]
        
        train_soundfiles = []
        test_soundfiles  = []
        
        for f in soundfiles:
            if f in db1.soundfile:
                train_soundfiles.append(f)
            if f in db2.soundfile:
                test_soundfiles.append(f)
            if f not in db1.soundfile and f not in db2.soundfile:
                raise Exception("Soundfile doesn't seem to belong to neither train nor test set")
        
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
    
    global mel_feat
    
    mel_feat = np.clip(np.nan_to_num(logfbank(signal, rate, lowfreq=500, 
                                              winlen=0.0232, winstep=0.0232,
                                           nfilt=settings.N_MEL_FILTERS)), 
                       a_min=-100, a_max=100)
    
    d, M = mel_feat.shape
    x = np.arange(d)

    feats = []
    bin_counter = 0
    for bin_ in mel_feat.T:
        fft_ = np.fft.fft(bin_)            
        f = interp1d(x, fft_)
        xnew = np.arange(10)
        fx = f(xnew)
        feats.append(fx)
        bin_counter += 1
    
    return np.hstack( feats )        
    
def extract_hertz(signal, rate, normalise):
    
    global hertz_feat, signal__
    
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
        
    global mel_feat, new_mel_feat
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
    rprint("AA[%d.%02d/%d] [pid:%d] analysing %s" % (soundfile_counter+1, chunk_counter, 
                              n_soundfiles, os.getpid(), os.path.basename(soundfile)) )
    
#    if highpass_cutoff > 0:
#        signal = usignal.highpass_filter(signal[:], cutoff=highpass_cutoff)
            
    
    if   analyser == "mfcc"           : feat = extract_mfcc(signal, rate, normalise)
    elif analyser == "mel-filterbank" : feat = extract_mel(signal, rate, normalise)
    elif analyser == "melfft"         : feat = extract_melfft(signal, rate, normalise)
    elif analyser == "hertz-spectrum" : feat = extract_hertz(signal, rate, normalise)
    elif analyser == "oskmeans"       : feat = extract_oskmeans(signal, rate, normalise)
    else:
        raise ValueError("Feature extraction method '%s' not known." % analyser)
    
    misc.dump_to_pickle(feat, picklename)
    
    return feat
