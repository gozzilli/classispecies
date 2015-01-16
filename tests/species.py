import sys, os
from os.path import expanduser as home
from tempfile import mkdtemp
import re
import gc

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.pylab import *

import subprocess
import shutil
huyrc = matplotlib.rc_params_from_file('/home/dz2v07/.config/matplotlib/matplotlibrc.huy')

sys.path = ['..'] + sys.path # this is where the features module is stored
from features import logfbank, fbank
import wavio

UKORTH_DIR = home("~/Dropbox/Shared/Orthoptera Sound App/species_recordings")
BL_DIR = home("/home/dz2v07/cicada-largefiles/bl-files/all/")

class NotAWaveFileError(Exception):
    def __init__(self, filename):
        self.filename = filename
    def __str__(self):
        return "%s is not a WAVE file" % self.filename
    
def make_axes2Dplot(data, signal, fs):
    x = linspace(0, len(signal)/float(fs), data.shape[1])
    y = linspace(0, fs/2, data.shape[0])
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

def downscale(data, time):
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
    data2 = data2[:len(t2)]
    return data2, t2

def downscale_spectrum(feat, target):
    ''' feat must be the right way up already (i.e. for fbank and the lot, transpose it first) '''

    out = []
    no_y, no_x = feat.shape
    
    newx = linspace(0, no_x, target)
    incr = no_x/float(target)
    
    for i in range(target):
        #print "A", feat[:,incr*i:incr*(i+1)]
        #print "B", mean(feat[:,incr*i:incr*(i+1)], axis=1)[np.newaxis]
        out.append( mean(feat[:,incr*i:incr*(i+1)], axis=1)[np.newaxis].T )

    return array(hstack(out))
        
    
    
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

def spectr_wave(filename, signal, fs, melfft, hertzfft, NFFT1, NFFT2):
    '''
    Small waveform, large spectrogram
    '''
    
    fig = plt.figure(figsize=(20,8), dpi=150)
    fig.subplots_adjust(hspace=0.05, wspace=0.09)
    
    #                       c,r    r,c
    ax1 = plt.subplot2grid((2,5), (1,0)) # waveform
    ax2 = plt.subplot2grid((2,5), (0,0)) # spectrogram
    ax3 = plt.subplot2grid((2,5), (1,1)) # mel, log
    ax4 = plt.subplot2grid((2,5), (0,1)) # mel, no log
    ax5 = plt.subplot2grid((2,5), (1,2)) # hertz, log
    ax6 = plt.subplot2grid((2,5), (0,2)) # hertz, no log
    ax7 = plt.subplot2grid((2,5), (1,3)) # hertz, 10, log
    ax8 = plt.subplot2grid((2,5), (0,3)) # hertz, 10, no log
    ax9 = plt.subplot2grid((2,5), (1,4)) # hertz, 10 left-most %, 10 bins, log
    ax10= plt.subplot2grid((2,5), (0,4)) # hertz, 10 left-most %, 10 bins, no log
    
    ### PLOT 1 - Waveform      
    t = np.linspace(0, len(signal)/float(fs), len(signal))
    
    # if necessary, downscale the waveform
    ax = ax1
    d_signal, d_time = downscale(signal, t)

    ax.tick_params(labelleft='off')
    ax.autoscale(tight=True)
    ax.set_ylabel("Amplitude", labelpad=22)
    ax.plot(d_time, d_signal, linewidth=0.2, rasterized=True)
    ax.set_xlabel("Time (s)")

    ### PLOT 2 - Spectrogram
    ax = ax2
    specgram = ax.specgram(signal, Fs = fs, scale_by_freq=True, rasterized=True)
    
    ax.autoscale(tight=True)           # no space between plot an axes
    #ax.get_xaxis().set_visible(False) # remove the x tick for top plot
    yticks = np.arange(0,fs/2,5000)    # |
    ax.set_yticks(yticks)              # |
    ax.set_yticklabels(yticks/1000)    # |
    ax.set_ylabel("Freq (kHz)")        # |> change Hz to kHz
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    ax.set_title("Spectrogram and waveform")
    
    if melfft.shape[1] > 500:
        melfft = downscale_spectrum(melfft, 500)
    if hertzfft.shape[1] > 500:
        hertz500 = downscale_spectrum(hertzfft, 500)
    else:
        hertz500 = hertzfft
    if hertzfft.shape[1] > 10:
        hertz10 = downscale_spectrum(hertzfft, 10)
    else:
        hertz10 = hertzfft
    
    hertz10_10 = hertzfft[:,0:hertzfft.shape[1]/10.]
    if hertz10_10.shape[1] > 10: 
        hertz10_10 = downscale_spectrum(hertz10_10, 10)
    
    #if LOG:
    #    melfft = 10*log10(melfft)
    #    hertzfft = 10*log10(hertzfft)

    stopfreq = float(fs)/NFFT1/2
    
    y = linspace(0, melfft.shape[0], melfft.shape[0]+1)
    x = linspace(0, stopfreq, melfft.shape[1]+1)
    
    ax = ax3
    _ = ax.pcolormesh(x, y, 10*log10(melfft), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    ax = ax4
    _ = ax.pcolormesh(x, y, melfft, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of mel spectrum (max 500 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    
    x = linspace(0, stopfreq, hertz500.shape[1]+1)
    y = linspace(0, fs/2/1000., NFFT1/2+1)
    
    ax = ax5
    _ = ax.pcolormesh(x, y, 10*log10(hertz500), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    ax = ax6
    _ = ax.pcolormesh(x, y, hertz500, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of hertz spectrum (max 500 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    x = linspace(0, stopfreq, hertz10.shape[1]+1)

    ax = ax7
    _ = ax.pcolormesh(x, y, 10*log10(hertz10), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    
    ax = ax8
    _ = ax.pcolormesh(x, y, hertz10, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of hertz spectrum (10 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    
    x = linspace(0, stopfreq/10., hertz10_10.shape[1])

    ax = ax9
    _ = ax.pcolormesh(x, y, 10*log10(hertz10_10), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Log-frequency (kHz)")
    
    
    ax = ax10
    _ = ax.pcolormesh(x, y, hertz10_10, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("10 leftmost % (max 10 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Frequency (kHz)")
    
    fig.suptitle(re.sub("\.wav", "", filename, flags=re.IGNORECASE), fontsize=16, y=0.98)

    del _
    gc.collect()
    # save
    #plt.show()
    #plt.close()
        
    # print "finished %s" % name
       
def pdfpages_savefig(filename):
    pdf.savefig()
    
def plt_savefig(filename):
    savefig(filename)

def allplot(wavdir, outfilename, savefig_, NFFT1, NFFT2, OVERLAP, counter=0, total=0):
    
    for soundfile in ['8kHz wave with 6Hz chirp.wav'] + sorted(os.listdir(wavdir)):

        counter += 1
        if not soundfile.lower().endswith(".wav"): continue

        sys.stdout.write("\r[%d/%d] %s" %(counter, total, outfilename % counter))
        sys.stdout.flush()

        try:

            if '8kHz wave with 6Hz chirp.wav' == soundfile:
                signal, fs, enc = open_audio(soundfile)
            else:
                signal, fs, enc = open_audio(os.path.join(wavdir, soundfile))

            # nfilt=40, winstep=0.0232, winlen=0.0232)[0]
            mel_feat = fbank(signal, nfilt=40, 
                             winlen=NFFT1/float(fs), 
                             winstep=(NFFT1-OVERLAP)/float(fs)
                             )[0]
            melfft = half2Darray(abs(scipy.fft(mel_feat.T, n=NFFT2)))

            # data, freqs, bins, _ = specgram(signal, Fs=fs)
            # plt.close()
            # hertzfft = half2Darray(abs(scipy.fft(data, n=NFFT)))

            frame_size = NFFT1 #samples
            hop        = NFFT1-OVERLAP #samples
            hertz_feat = half2Darray(abs(stft2(signal, fs, frame_size, hop))).T
            hertz_fft  = half2Darray(np.abs(scipy.fft(hertz_feat, n=NFFT2)))

            spectr_wave(soundfile, signal, fs, melfft, hertz_fft, NFFT1, NFFT2)
            savefig_(outfilename % counter)  # saves the current figure into a pdf page
            close()
        except MemoryError:
            print "\nmemory error on %s" % soundfile
            mel_feat = melfft = hertz_feat = hertz_fft = None
            close() 
            raise

    
WAVDIR_OPTS = [#('uk-orthoptera_2xFFT-%02d-all.pdf', UKORTH_DIR), 
               ('bl-orthoptera_2xFFT-%02d-all.pdf', BL_DIR)
               ]
POWER_OPTS  = [7, 14, 16]

counter = 0
total = sum([len(os.listdir(wavdir)) for _, wavdir in WAVDIR_OPTS]) * len(POWER_OPTS)

for pdffilename, wavdir in WAVDIR_OPTS:
    for nfft_power in POWER_OPTS:

        try:
            NFFT1 = 256
            OVERLAP = 0
            NFFT2 = 2**nfft_power
        
        
            tempdir = mkdtemp()
            print tempdir
            outfilename = os.path.join(tempdir, "temp_%04d.pdf")
            allfilename = pdffilename % NFFT2
        
        
            allplot(wavdir, outfilename, plt_savefig, NFFT1, NFFT2, OVERLAP, counter=counter, total=total)
        
        finally:
                
            reducedfilename = allfilename[0:-4]+"-reduced.pdf"
            #!pdftk $tempdir/*.pdf cat output $allfilename
            _exit_code = subprocess.call("pdftk %s/*.pdf cat output %s" %(tempdir, allfilename), shell=True) 
            if _exit_code == 0:
                print "Output in", allfilename 
                #!gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$reducedfilename
                _exit_code = subprocess.call("gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=%s %s" % (reducedfilename, allfilename), shell=True)
                if _exit_code == 0:
                    print "Reduced output in", reducedfilename 
                    shutil.rmtree(tempdir)
            else:
                print "PDFTK failed"
     