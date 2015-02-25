# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:44:16 2014

@author: Davide Zilli
"""

from __future__ import print_function

import os
import traceback

import numpy as np
import scipy
import scipy.io.wavfile as wav
import scipy.fftpack

from scipy.cluster.vq import  whiten

from features import mfcc, logfbank, fbank
from stowell.oskmeans import OSKmeans

from multiprocessing.pool import ApplyResult
from multiprocessing import Pool, cpu_count


import settings
from utils import misc, signal as usignal
from utils.misc import rprint, deprecated
from utils.plot import file_plot

print (os.path.abspath('.'))
logger = misc.config_logging("classispecies")

class FeatureSet(object):
    
    data        = None
    labels      = None
    max_length  = None
    min_length  = None
    avg_length  = None
    tot_length  = None
    db          = None
    type        = None
    
    def __init__(self, type_, an):
        
        assert type_ in ["train", "test"], "Type must be either train or test"
        
        self.type = type_
        self.an   = an.replace(" ", "-") 
    
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
        
        missingLabels = False
        if labels == None:
            labels = np.empty( len(soundfiles) )
            missingLabels = True
            
        new_labels = []
        res = []
        
        soundfile_counter = 0
        total_counter     = 0
        chunk2soundfile   = []
        db = []
        
        allpicklename = misc.make_output_filename("all-%s-%s" % (self.type, self.an), settings.analyser+str(settings.sec_segments or ""),
                                                       settings.modelname, "pickle", removeext=False)
                                                       
        if not os.path.exists(allpicklename) or settings.FORCE_FEATXTRALL:
            
            if settings.MULTICORE:
                logger.info ("Starting pool of %d processes" % (cpu_count()))
                pool = Pool(processes=cpu_count())
                
            picklenames = []
    
            for soundfile, lab in zip(soundfiles, labels):
            
    #            logger.info( "[%d.00/%d] analysing %s" % (soundfile_counter+1, len(soundfiles), os.path.basename(soundfile) ), end="" )
    #            sys.stdout.flush()
    
                #settings.FEATURE_ONEFILE_PLOT = 0 == soundfile_counter # only the first time
                
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
                    #signals = np.array([signal_all])
                    signals = [signal_all]
                    
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
                    
                    assert settings.agg != None, "agg should not be none"
                    params = "%s%s%s%s%s" % ("mel" if settings.extract_mel else "hertz",
                                                     "-log" if settings.extract_dolog else "",
                                                     "-dct" if settings.extract_dct else "",
                                                     "-%s" % settings.agg,
                                                     "-%.1fsec" % settings.sec_segments if settings.sec_segments else "-entire")
                    
                    picklename = misc.make_output_filename("%s-%s" % (chunk_name, params), settings.analyser+str(settings.sec_segments or ""),
                                                           settings.modelname, "npy", removeext=False)
                    picklenames.append(picklename)
                    
                    if not os.path.exists(picklename) or settings.FORCE_FEATXTR:
                        
                        is_analysing = True
                        
                        if settings.MULTICORE:
                            res.append( pool.apply_async(exec_featextr, 
                            #x = pool.apply_async(exec_featextr,
                            #pool.apply_async(exec_featextr,
                                            [soundfile, signal, rate, analyser, picklename,
                                             soundfile_counter, chunk_counter, len(soundfiles), 
                                             highpass_cutoff, normalise])
                            )
                            #x.get()
                        else:
                            #res.append( exec_featextr(soundfile, signal, rate, analyser, picklename,
                            exec_featextr(soundfile, signal, rate, analyser, picklename,
                                             soundfile_counter, chunk_counter, len(soundfiles), 
                                             highpass_cutoff, normalise) 
                            #)
                          
                    else:
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
                
            if settings.MULTICORE:
                pool.close()
                pool.join()
                
            ## check for exceptions
            
            if settings.MULTICORE:
                for r in res:
                    try:
                        r.get()
                    except:
                        print (self.an)
                        raise
                    
            res = []
            
            
            print ()                
            logger.debug ("unpickling %d files" % len(picklenames))
            for picklename in picklenames:
                try:
                    res.append(misc.load_from_npy(picklename))
                except EOFError as e:
                    print ("EOFError on {}".format(picklename))
                    raise e
            
                
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
    

#             print (len(X))
            print ("X[0].shape:", X[0].shape)
#             for x in X:
#                 print (x.shape, end=" ")             
            try:
                X = np.vstack(X)
            except ValueError as e: 
                print (e)
                for x in X:
                    print (x.shape)
                raise
            new_labels = np.vstack(new_labels)
            
            #print()
            #print ("multilabel", settings.MULTILABEL)
            if settings.MULTILABEL:
                if missingLabels:
                    truth_dtype = "1b"
                else:
                    truth_dtype = "%db" % labels.shape[1]
            else:
                truth_dtype = "S5"
            
            #print (str(labels.shape))
            #print (truth_dtype)
            
            #self.db_ = db[:]
            
            #print (db[-1])
            
            db = np.array(db, dtype=[("soundfile"        , 'S200'), 
                                     ("chunk_counter"    , int),
                                     ("length"           , int),
                                     ("truth"            , truth_dtype)]).view(np.recarray)
            
            #misc.dump_to_pickle( (X, new_labels, db), allpicklename)
        
        else:
            X, new_labels, db = misc.load_from_pickle(allpicklename)

        logger.debug("\nX.shape[0]: %d, new_labels.shape[0]: %d" % (X.shape[0], new_labels.shape[0]))
        assert X.shape[0] == new_labels.shape[0]
        logger.info("")

        
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
        
        np.random.seed(settings.RANDOM_SEED)
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
        
        logger.debug ("DBs train: %d, test: %d" %( len(db1.soundfile), len(db2.soundfile) ))
        
        missing = []
        
        for f in soundfiles:
            if f in db1.soundfile:
                train_soundfiles.append(f)
            if f in db2.soundfile:
                test_soundfiles.append(f)
            if f not in db1.soundfile and f not in db2.soundfile:
                logger.warning ("%s missing" % f)
                missing.append(f)
                #raise Exception("Soundfile %s doesn't seem to belong to neither train nor test set" % f)
        
        if missing:
            logger.warning ("missing (%d files):\n%s" % (len(missing), missing))
        
        logger.debug ("train soundfiles length: %d" % len(train_soundfiles))
        logger.debug ("test  soundfiles length: %d" % len(test_soundfiles))
        
        fs1 = FeatureSet("train", self.an)
        fs1.db = db1
        fs1.data = data1
        fs1.labels = lab1
        fs1.set_stats()
        
        fs2 = FeatureSet("test", self.an)
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
    
    

def aggregate_feature(feat, normalise=False, axis=0):
    ''' Take any combination of mean, std and max of the given features
    
    according to the parameters in settings. 
    
    Args
        feat             -- the 2D array of features
        normalise (bool) -- normalise or not?
        
    Return
        a horizontally stacked vector (1-row array) such as:
        [mean1 mean2 mean3 var1 var2 var3 max1 max2 max3]
        
    '''
    
    if axis == None:
        raise Exception("Axis cannot be None")
    
    out_feat = []
    func = {np.nanmean:settings.extract_mean,
            np.nanstd:settings.extract_std,
            np.nanmax:settings.extract_max}
    func = [x for x in func.keys() if func[x]]
    
    if normalise:
        feat = usignal.rms_normalise(feat)

    for f in func:
        
        #print ("\nusing {}\n".format(f.func_name))
        
        r = f(feat, axis=axis)
        if settings.whiten_feature:
            r = whiten(r)
        out_feat.append(r)
        
    if not out_feat: ## if still empty, take the entire feature set
        out_feat = feat
        
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

@deprecated
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
        
    return np.hstack(10*np.log10(usignal.downscale_spectrum(melfft, 10)))
    
    
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

@deprecated
def extract_hertzfft(signal, rate, normalise):
    
    #NFFT = 2**14
    NFFT = 128
    
    frame_size = settings.NFFT1 #samples
    hop        = settings.NFFT1-settings.OVERLAP #samples

    hertz_feat = usignal.half2Darray(np.abs(usignal.stft_bysamples(signal, rate, frame_size, hop))).T
    hertz_fft  = usignal.half2Darray(np.abs(scipy.fft(hertz_feat, n=NFFT)))
    
    feat = 10*np.log10(#usignal.downscale_spectrum(
                           usignal.downscale_spectrum(hertz_fft, 10, axis=1), 
                       #10, axis=0)
                      )
    
    if settings.FEATURE_ONEFILE_PLOT:
        file_plot(signal, rate, feat)
        settings.FEATURE_ONEFILE_PLOT = False
        
    agg = np.hstack(feat)
    return agg




@deprecated    
def extract_hertz(signal, rate, normalise):
    
    signal__ = signal
#        hertz_feat = usignal.stft(signal, rate, 0.1, 0.01)        
#        hertz_feat = 10*np.log10(hertz_feat.T[:len(hertz_feat.T)/2])
#        hertz_feat = np.clip(np.nan_to_num(hertz_feat), a_min=-100, a_max=100)
 
    frame_size = 0.0018 # for 1sec of signal, 0.0005 = 22 bins, then divided by two 11 (22)
    hop = 0.003 # also noverlap in specgram, roughly every 128 samples

    hertz_feat = 10*np.log10(usignal.stft_byseconds(signal, rate, frame_size, hop))
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


def extract_multiple(signal, rate, normalise, soundfile):
    
    normalise = settings.normalise
    
    if settings.extract_mel:
        feat, energy = fbank(signal, samplerate=rate,nfilt=40, 
                     winlen=settings.NFFT1/float(rate), 
                     winstep=(settings.NFFT1-settings.OVERLAP)/float(rate),
                     lowfreq=settings.highpass_cutoff
                     )
    else:    
        frame_size = settings.NFFT1         #samples
        hop        = settings.NFFT1-settings.OVERLAP #samples
        #feat = usignal.half2Darray(np.abs(usignal.stft_bysamples(signal, rate, frame_size, hop)))
        feat = usignal.stft_bysamples_optimised(signal, rate, frame_size, hop)
        #print ("feat shape", feat.shape)
        assert feat.shape[1] == 32
        #print (feat.shape)
        filter_ = int(settings.highpass_cutoff/(rate)*settings.NFFT1)
        feat = feat[:,filter_:]
        #print ("AAA\n", feat.shape)
        
    if normalise:
        logger.debug("normalising")
        feat = usignal.rms_normalise(signal, feat)
        
    if settings.extract_dolog:
        feat = np.clip(np.log(feat), -100, 100)
    
    if settings.extract_dct:
        appendEnergy = True

        dct_ = scipy.fftpack.dct(feat, type=2, axis=1, norm='ortho')[:,:settings.NMFCCS]
        
        ## lifter
        n = np.arange(settings.NMFCCS)
        lift = 1+ (settings.ceplifter/2)*np.sin(np.pi*n/settings.ceplifter)
        dct_ = lift*dct_
        
        if appendEnergy and settings.extract_mel: dct_[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy                    
        
        feat = dct_
        
    if settings.extract_fft2:
        
        feat = usignal.half2Darray(np.abs(scipy.fft(feat, axis=0)), axis=0)
        
        if settings.MOD_TAKE1BIN:
            feat = feat[0,:]
        else:
        
            _shape_orig = feat.shape
            
            rate = float(rate)
            secs  = len(signal)/rate
            rate2 = rate/settings.NFFT1/2.
            upper = settings.mod_cutoff * secs # number of samples in the first 50 Hz of the MOD
            
            if settings.extract_logmod:
                feat = usignal.log_mod(feat, rate, settings.NFFT1)
            else:
                feat = usignal.downscale_spectrum(
        #                     usignal.downscale_spectrum(
                                feat[0:upper,:],
        #                     settings.downscale_factor, axis=1),
                       settings.downscale_factor, axis=0)
                #print ("\ndownscaled by {}, feat size: {}x{} to {}x{}".format(settings.downscale_factor, *(_shape_orig+feat.shape)), end="")
            
            #feat = feat[0:8192,:]

        
    # TODO do consistency tests here
    
#     print ("before Transpose feature is {}".format(feat.shape))
    feat = feat.T
    
    if settings.FEATURE_ONEFILE_PLOT:
#         np.savetxt('/tmp/featall.txt', feat)
#         return
        #print ("\nfeat shape: {}".format(feat.shape))
        
        #print ("upper: %s" %upper)
        file_plot(signal, rate, feat, soundfile)
        settings.FEATURE_ONEFILE_PLOT = False
        #np.savetxt('/tmp/featagg.txt', feat[:,0])
    
    agg = aggregate_feature(feat, axis=1)
    #agg = np.hstack( (feat[:,0], mean_)  )
    #agg = feat[:,0]
    #agg = np.hstack( (mean_, feat[:,0]) )
#     agg = mean_
#     try:
#         assert np.all(np.hstack( (std_, mean_) ) == agg)
#     except AssertionError:
#         print ("%s\n\n%s" % (np.hstack( (std_, mean_) ), agg))
#         np.savetxt('/tmp/1.txt', np.hstack( (std_, mean_)))
#         np.savetxt('/tmp/2.txt', agg)
#         raise
    print ("feature is {}".format(agg.shape), end="")

    return agg
    

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
        elif analyser == "multiple"       : feat = extract_multiple(signal, rate, normalise, soundfile)
        else:
            raise ValueError("Feature extraction method '%s' not known." % analyser)
        
        misc.dump_to_npy(feat, picklename)
        
        return feat
    except:
        print (soundfile, misc.get_an())
        #traceback.print_exc()
        raise
        
