import os

import scipy
from scipy.io import wavfile

import numpy as np

import wavio
lengths = []
fss     = []

class NotAWaveFileError(Exception):
    def __init__(self, filename):
        self.filename = filename
    def __str__(self):
        return "%s is not a WAVE file" % self.filename
    
def make_axes2Dplot(data, signal, fs):
    ''' Create the axes for the plot of a 2D matrix.
    
    `pcolormesh` takes either the matrix alone or x,y,matrix, where x and y
    are arrays of values for the x and y axes. Note that the length of x and y
    should be one more than the dimension of the matrix (e.g., for a 40x40
    matrix, x and y should be 41 elements each).
    '''
    
    x = np.linspace(0, len(signal)/float(fs), data.shape[1])
    y = np.linspace(0, fs/2, data.shape[0])
    
    return x,y
    
    
def stats(data, fs):
    fs = float(fs)
    nsecs = len(data)/fs
    lengths.append(nsecs)
    fss.append(fs)


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

def downscale_spectrum(feat, target):
    ''' feat must be the right way up already (i.e. for fbank and the lot, transpose it first) '''

    out = []
    no_y, no_x = feat.shape
    
    newx = np.linspace(0, no_x, target)
    incr = no_x/float(target)
    
    for i in range(target):
        #print "A", feat[:,incr*i:incr*(i+1)]
        #print "B", mean(feat[:,incr*i:incr*(i+1)], axis=1)[np.newaxis]
        out.append( np.mean(feat[:,incr*i:incr*(i+1)], axis=1)[np.newaxis].T )

    return np.array(np.hstack(out))
        
    
    
''' OLD DOWNSCALE SPECTRUM     
    x = range(no_x)
    bin_counter = 0
    
    for bin_ in feat:
    
        f = interp1d(x, bin_)
        xnew = np.arange(target)
        fx = f(xnew)
        feats.append(fx)
        bin_counter += 1
        
    return np.array(feats)
'''

def half2Darray(arr):
    return arr[:,:arr.shape[1]/2]

def stft_byseconds(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X


def stft_bysamples(x, fs, framesamp, hopsamp):
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X
