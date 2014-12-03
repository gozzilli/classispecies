# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:44:16 2014

@author: Davide Zilli
"""

from __future__ import print_function

import sys, os
import numpy as np
from scipy.interpolate import interp1d
from scipy.cluster.vq import  whiten

from features import mfcc, logfbank, fbank
from stowell.oskmeans import OSKmeans, normalise_and_whiten

import settings
from utils import misc, signal as usignal


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
    print( "\r[%d.%02d/%d] [pid:%d] analysing %s\r" % (soundfile_counter+1, chunk_counter, 
                              n_soundfiles, os.getpid(), os.path.basename(soundfile)), end="" )
    sys.stdout.flush()
    
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
