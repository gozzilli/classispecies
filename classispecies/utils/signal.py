import numpy as np
import scipy
import scipy.signal

def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X


def stft2(x, fs, framesamp, hopsamp):
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X
    
# def stft(x, fs, framesz, hop):
#     framesamp = len(x)
#     hopsamp = int(hop*fs)
#     w = scipy.hamming(framesamp)
#     X = scipy.array(abs(scipy.fft(w*x, n=128)) )
#     return X


def rms_normalise(data):
    ''' return a signal, normalised by its root-mean-sqaure value
    
    Args:
        data:   the signal (generally the spectrogram of the audio signal)
    '''
    
    _rms = np.sqrt(np.mean(np.square(data)))
    return data*_rms

def highpass_filter(data, fs=44100, cutoff=2000., filterorder=1):
    ''' butterworth high-pass filter '''
    
    b,a = scipy.signal.filter_design.butter(filterorder,float(cutoff)/fs/2., btype="high")
    return scipy.signal.lfilter(b,a,np.abs(data))