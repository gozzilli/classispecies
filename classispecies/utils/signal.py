import os
import numpy as np
import scipy.signal
from scipy.io import wavfile
import wavio


def half2Darray(arr, axis=1):
    assert axis in [0,1], "axis parameter can only be 0 or 1"
    
    if 0 == axis:
        return arr[:arr.shape[0]/2,:]
    else:
        return arr[:,:arr.shape[1]/2]

def stft_byseconds(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    
    return stft_bysamples(x, fs, framesamp, hopsamp)


def stft_bysamples(x, fs, framesamp, hopsamp):
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def stft_bysamples_optimised(x, fs, framesamp, hopsamp):
    
    w = scipy.hamming(framesamp)
    bins = range(0, len(x)/hopsamp)
    X = np.zeros((len(bins), framesamp/2))
    
    for i in bins:
        inc = i*hopsamp
        X[i,:] = np.abs(scipy.fft(w*x[inc:inc+framesamp]))[0:framesamp/2]
        
    return X


def rms_normalise(signal, feat):
    ''' return a signal, normalised by its root-mean-square value
    
    Args:
        data:   the signal (generally the spectrogram of the audio signal)
    '''
    
    _rms = np.sqrt(np.nanmean(np.square(signal)))
    return feat/_rms


def highpass_filter(data, fs=44100, cutoff=2000., filterorder=1):
    ''' butterworth high-pass filter '''
    
    b,a = scipy.signal.filter_design.butter(filterorder,float(cutoff)/fs/2., btype="high")
    return scipy.signal.lfilter(b,a,np.abs(data))

def downscale(data, time, normalise=False):
    data2 = []
    t2 = []
    q = 100
    r = 2*q
    for i in range(0,len(data),r):
        data2.append(max(data[i:i+r]))
        data2.append(min(data[i:i+r]))
        try:
            t2.append(time[i+int(0.25*r)])
            t2.append(time[i+int(0.75*r)])
        except:
            #print i+int(0.75*r)
            pass
    #plt.gca().get_xaxis().set_visible(False)
    data2 = np.array(data2[:len(t2)])
    t2    = np.array(t2)
    
    if normalise:
        data2 = data2/float(np.max(np.max(data2)))
    return np.array(data2), np.array(t2)

def downscale_spectrum(feat, target, axis=1):
    ''' Down-scale a NxM matrix to a set number of bins (by averaging)
    
    Param:
        feat (np.array): the matrix. It must be the right way up already (i.e. for fbank 
              and the lot, transpose it first) 
        target (int): the number of bins to down-scale to
    
    Return:
        the down-scaled array, or the input array if `target` is smaller than 
        the axis to be down-scaled (i.e. axis 1, or the x axis, or the columns)
    '''
    

    axis_len = feat.shape[axis]
    
    if axis_len <= target:
        return feat
    
    incr = axis_len/float(target)
    
    
    if axis == 1:
        out = np.ndarray ( (feat.shape[0], target) )
        for i in range(target):
            out[:,i] = np.mean(feat[:,incr*i:incr*(i+1)], axis=1).T 
        
        
    else:
        out = np.ndarray ( (target, feat.shape[1]) )
        for i in range(target):
            out[i,:] = np.mean(feat[incr*i:incr*(i+1),:], axis=0) 
        #return np.array(np.vstack(out))
    return out
    


class NotAWaveFileError(Exception):
    def __init__(self, filename):
        self.filename = filename
    def __str__(self):
        return "%s is not a WAVE file" % self.filename

def open_audio(filename):
    '''
    open audio file with scipy or audiolab, depending on what is available.
    Return data, fs, enc
    '''
    
    if not filename.lower().endswith(".wav"):
        raise NotAWaveFileError(filename)
    try:
        fs, data = wavfile.read(filename)
    except ValueError:
        fs, sampwidth, data = wavio.readwav(filename)
        print "\t%s is a %dbit audio file" % (os.path.basename(filename), 8*sampwidth)
        
    enc = None
    #data, fs, enc = audiolab.wavread(sound) # same with audiolab
        
    # if the audio sample is stereo, take only one channel. May not work as
    # desired if the two channels are considerably different. 
    if len(data.shape) > 1:
        data = data[:,1]
    
    return (data, fs, enc)

def log_mod(mod, fs, nfft, nbins=48):
    
    # Sonogram

    sonogramSamplingRate = fs / nfft
    
    # FFT
    
    fftMaximumFrequency  = sonogramSamplingRate / 2.0
    fftLength = mod.shape[0]
    fftFrequencies = np.linspace(0, fftMaximumFrequency, fftLength)
    fftFrequencies = fftFrequencies[1:]
    
    
    # Calculate binning range
    
    lowestLogFrequency = np.log10(fftMaximumFrequency / fftLength)  # This can change. Lowest frequency that might be interesting.
    highestLogFrequency = np.log10(fftMaximumFrequency)             # Always the maximum frequency
    
    binFrequencies = np.power(10.0, np.linspace(lowestLogFrequency, highestLogFrequency, nbins))
    
    # Calculate bin mapping
    
    interpolatedBins = np.int32(np.round(np.interp(
      binFrequencies, 
      fftFrequencies, 
      np.linspace(1, fftLength-1, fftLength-1)
    )))
    
    mod_interpd = np.ones( (nbins, mod.shape[1]) )
    mod_interpd[0,:] = mod[0,:]
    
    for x in range(1,nbins):
        
        if interpolatedBins[x-1] == interpolatedBins[x]:
            z = mod[interpolatedBins[x-1],:]
        else:
            z = np.nan_to_num(np.nanmean(mod[interpolatedBins[x-1]+1:interpolatedBins[x]+1,:], axis=0))
            
        mod_interpd[x,:] = z
        
    assert np.all(mod[0,:] == mod_interpd[0,:]), "The log mod is missing the original first bin."
        
    return mod_interpd

     
### OLD DOWNSCALE SPECTRUM  
"""
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