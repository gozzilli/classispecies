# -*- coding: utf-8 -*-

import sys
sys.path = ['..'] + sys.path

import os.path
import re
import pandas as pd

import numpy as np
import scipy.fftpack

import matplotlib
matplotlib.use('pdf')
from classispecies.utils.misc import load_from_npy, mybasename,\
    get_output_filename
from classispecies.utils.signal import downscale, downscale_spectrum, half2Darray, stft_bysamples, open_audio, stft_bysamples_optimised, log_mod

from matplotlib import pyplot as plt

from features import fbank, mfcc

huyrc = matplotlib.rc_params_from_file('/home/dz2v07/.config/matplotlib/matplotlibrc.nocomment.huy')
#matplotlib.rcParams.update(huyrc)
matplotlib.rc('font', family='Palatino Linotype')

from classispecies import settings
settings.OUTPUT_DIR = '../outputs2'

def make_suptitle(fig, datum, filename):
    fontsize=20 
    y=0.97
    if isinstance(datum, pd.Series):
        fig.suptitle(re.sub(r'[ ]+,', ',', "%s %s %s %s %s" % (datum.species_eng if not pd.isnull(datum.species_eng) else "undefined", 
                                    "(%s)" % datum.species_latin if not pd.isnull(datum.species_latin) else "",
                                    ", %s" % datum.family_latin if not pd.isnull(datum.family_latin) else "",
                                    "- %s" % datum.location if not pd.isnull(datum.location) else "",
                                    ", %s" % int(datum.year) if not pd.isnull(datum.year) else "",
                                    #"[%s]" % datum.sound if not pd.isnull(datum.sound) else ""
                                    )), y=y, fontsize=fontsize)
        plt.text(1.0, 1.2, datum.sound,
            horizontalalignment='right',
            verticalalignment='top',
            transform=plt.gca().transAxes,
            fontsize=9)
    else:
        fig.suptitle(re.sub("\.wav", "", os.path.basename(filename).replace("_", " "), flags=re.IGNORECASE), y=y, fontsize=fontsize)

def gridplot1(filename, signal, fs, melfft, hertzfft, NFFT):
    '''
    Small waveform, large spectrogram
    Deprecated. Use gridplot2 instead
    '''
    
    LOG = False
    
    fig = plt.figure(figsize=(25,5), dpi=150)
    fig.subplots_adjust(hspace=0.2,)
    
    ax1 = plt.subplot2grid((6,4), (4,0), rowspan=2) # waveform
    ax2 = plt.subplot2grid((6,4), (0,0), rowspan=4) # spectrogram
    ax3 = plt.subplot2grid((6,4), (0,1), rowspan=6) # FFT of mel spectrum
    ax4 = plt.subplot2grid((6,4), (0,1), rowspan=6) # FFT of mel spectrum
    ax5 = plt.subplot2grid((6,4), (0,3), rowspan=6) # FFT of hertz spectrum, scaled to 500
    ax6 = plt.subplot2grid((6,4), (0,4), rowspan=6) # FFT of hertz spectrum, scaled to 10
    
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
    #if hertzfft.shape[1] > 500:
    #    hertzfft = downscale_spectrum(hertzfft, 500)    
        
    if LOG:
        melfft = 10*np.log10(melfft)
        hertzfft = 10*np.log10(hertzfft)

    
    ax = ax3
    ax.pcolormesh(melfft, rasterized=True)
    ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of mel spectrum")
    
    ax = ax4
    ax.pcolormesh(hertzfft, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of hertz spectrum")    
    
    ax = ax5
    ax.pcolormesh(downscale_spectrum(hertzfft, 500) , rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of hertz spectrum, downscaled to 500 bins")
    
    ax = ax6
    ax.pcolormesh(downscale_spectrum(hertzfft, 10), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of hertz spectrum, downscaled to 10 bins")
    
    fig.suptitle(filename, fontsize=14, y=1.02)

    # save
    #plt.show()
    #plt.close()
        
    # print "finished %s" % nam    
def gridplot2(filename, signal, fs, melfft, hertzfft, NFFT1, NFFT2, datum=None):
    '''
    Small waveform, large spectrogram
    '''
    
    fig = plt.figure(figsize=(27,10), dpi=150)
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
    
    y = np.linspace(0, melfft.shape[0], melfft.shape[0]+1)
    x = np.linspace(0, stopfreq, melfft.shape[1]+1)
    
    ax = ax3
    _ = ax.pcolormesh(x, y, 10*np.log10(melfft), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    ax = ax4
    _ = ax.pcolormesh(x, y, melfft, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of mel spectrum (max 500 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    
    x = np.linspace(0, stopfreq, hertz500.shape[1]+1)
    y = np.linspace(0, fs/2/1000., NFFT1/2+1)
    
    ax = ax5
    _ = ax.pcolormesh(x, y, 10*np.log10(hertz500), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    ax = ax6
    _ = ax.pcolormesh(x, y, hertz500, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of hertz spectrum (max 500 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    x = np.linspace(0, stopfreq, hertz10.shape[1]+1)

    ax = ax7
    _ = ax.pcolormesh(x, y, 10*np.log10(hertz10), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    
    ax = ax8
    _ = ax.pcolormesh(x, y, hertz10, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("FFT of hertz spectrum (10 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    
    x = np.linspace(0, stopfreq/10., hertz10_10.shape[1])

    ax = ax9
    _ = ax.pcolormesh(x, y, 10*np.log10(hertz10_10), rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_xlabel("Chirp frequency (Hz)")
    
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Log-frequency (kHz)")
    
    
    ax = ax10
    _ = ax.pcolormesh(x, y, hertz10_10, rasterized=True)
    #ax.set_xlim( (0,NFFT/2) )
    ax.autoscale(tight=True)
    ax.set_title("leftmost 10% (max 10 bins)")
    ax.tick_params(labelbottom='off') # labels along the bottom edge are off
    
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Frequency (kHz)")
    
    make_suptitle(fig, datum, filename)

    # save
    #plt.show()
    #plt.close()
    
    return fig

def gridplot3(filename, NFFT1, NFFT2, OVERLAP, datum):
    
    BG_ON_FEATURES = False
    BG_ON_WAVEFORM = False
    PLOT_WAVEFORM  = False
    
    signal, fs, enc = open_audio(filename)
        
    mel_spectrum, energy = fbank(signal, samplerate=fs,nfilt=40, 
                     winlen=NFFT1/float(fs), 
                     winstep=(NFFT1-OVERLAP)/float(fs)
                     )
    
    frame_size = NFFT1 #samples
    hop        = NFFT1-OVERLAP #samples
    hertz_spectrum = half2Darray(np.abs(stft_bysamples(signal, fs, frame_size, hop)))
    
    fig = plt.figure(figsize=(20,12), dpi=150)
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
        
    gridsize = (5,5)
    numcep = 13
    stopfreq_fft2 = float(fs)/NFFT1/2
    
    if BG_ON_FEATURES:            
        ax = plt.subplot2grid( gridsize, (2, 4))
        rec = plt.Rectangle((-4.8,-3.5),6,4.7,fill=True, color="green", alpha=0.1, lw=0)
        rec = ax.add_patch(rec)
        rec.set_clip_on(False)
        ax.axis('off')
        
    if PLOT_WAVEFORM:
    
        ### PLOT 1 - Waveform   
        row = 0
        col = 4
        t = np.linspace(0, len(signal)/float(fs), len(signal))
             
        # if necessary, downscale the waveform
        ax = plt.subplot2grid( gridsize, (row, col)) # 
        d_signal, d_time = downscale(signal, t, normalise=True)
             
        ax.autoscale(tight=True)
        ax.set_ylabel("Amplitude", labelpad=22)
        ax.plot(d_time, d_signal, linewidth=0.2, rasterized=True)
        ax.set_xlabel("Time (s)")
        
        if BG_ON_WAVEFORM:
        
            rec = plt.Rectangle((-0.2,np.min(d_signal)-0.6),np.max(d_time)+0.3,np.max(d_signal)*2+1.0,fill=True, color="yellow",lw=0, alpha=0.2)
            rec = ax.add_patch(rec)
            rec.set_clip_on(False)
    
    
    col = 4
    row = 1
    
    feat = mfcc(signal, samplerate=fs,nfilt=40, 
                     winlen=NFFT1/float(fs), 
                     winstep=(NFFT1-OVERLAP)/float(fs),
                     numcep=numcep).T
    t1 = np.linspace(0, len(signal)/float(fs), feat.shape[1]+1)
    ycep = np.linspace(0, numcep, numcep+1)
    
    ax = plt.subplot2grid( gridsize, (row, col))
    _ = ax.pcolormesh(t1, ycep, feat, rasterized=True)
    _ = ax.autoscale(tight=True)
    _ = ax.set_title("mfcc [{}x{}]".format(*feat.shape))
    
    x = np.linspace(0, fs/2./1000, feat.shape[0])
    row = 3
    max_ = np.max(feat, axis=1)
    ax = plt.subplot2grid( gridsize, (3, col)) # 
    _ = ax.plot(x, max_)
    _ = ax.autoscale(tight=True)
    _ = ax.set_title("max [%d]" % len(max_))
    _ = ax.set_ylabel("Magnitude") if col == 0 else None
    _ = ax.set_xlabel("Freq (kHz)")
    
    row = 4
    mean_ = np.mean(feat, axis=1)
    std_ = np.std(feat, axis=1)
    ax = plt.subplot2grid( gridsize, (4, col)) # 
    _ = ax.plot(x, mean_)
    _ = ax.plot(x, std_)
    _ = ax.autoscale(tight=True)
    _ = ax.set_title(u"$\mu$, $\sigma$ [%d]" % len(mean_))
    _ = ax.set_ylabel("Magnitude") if col == 0 else None
    _ = ax.set_xlabel("Freq (kHz)")
    
    
    for mel in [True, False]:
        
        for log in [True, False]:
            
            row = 0
            col = int(log)+2*int(mel)
    
            for dct in [True, False]:
                
                if mel:
                    feat = mel_spectrum.copy()
                else:
                    feat = hertz_spectrum.copy()
    
    
                t  = np.linspace(0, len(signal)/float(fs), len(signal))
                t1 = np.linspace(0, len(signal)/float(fs), feat.shape[0]+1) # time for time-freq spectra
                f1 = np.linspace(0, fs/2./1000, feat.shape[1]+1)                 # freq fft1 (0, 22050, 128)
                def f2(myfeat):
                    return np.linspace(0, stopfreq_fft2, myfeat.shape[1]+1)         # freq fft2 (0, 86, 64)
                ycep = np.linspace(0, numcep, numcep+1)
                
                if log:
                    feat = np.clip(np.log(feat), -100, 100)
                    
                if dct:
                    
                    row = 1
                    appendEnergy = True
                    ceplifter = L = 22
                    
                    #feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
                    #feat = numpy.log(feat)
                    dct_ = scipy.fftpack.dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
                    
                    ## lifter
                    nframes,ncoeff = np.shape(dct_)
                    n = np.arange(ncoeff)
                    lift = 1+ (L/2)*np.sin(np.pi*n/L)
                    dct_ = lift*dct_
                    
                    if appendEnergy and mel: dct_[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy                    
                    
                    dct_ = dct_.T
                    
                    ax = plt.subplot2grid( gridsize, (row, col)) # 
                    _ = ax.pcolormesh(t1, ycep, dct_, rasterized=True)
                    _ = ax.autoscale(tight=True)            
                    _ = ax.set_title(make_title(mel, log, dct, dct_))
                    _ = ax.set_xlabel("Time (s)")
                    _ = ax.set_ylabel("Freq coefficient") if col == 0 else None
                    
                    mod_ = dct_
                    
                else:
                    row = 2  
        
                    mod_ = half2Darray(np.abs(scipy.fft(feat, n=NFFT2, axis=0)).T)
                    
                    ax = plt.subplot2grid( gridsize, (row, col)) # 
                    _ = ax.pcolormesh(f2(mod_), f1, mod_, rasterized=True)
                    _ = ax.autoscale(tight=True)            
                    _ = ax.set_title(make_title(mel, log, dct, mod_))
                    _ = ax.set_xlabel("Modulation freq (Hz)")
                    _ = ax.set_ylabel("Freq (kHz)") if col == 0 else None
            
            feat = feat.T
            row = 0
                
            if mel: 
                #y = hz2mel(linspace(0, fs/2., feat.shape[0]+1))/1000.
                y = np.linspace(0, feat.shape[0], feat.shape[0]+1)
            else:
                y = f1
            
            #                      c,r        r,c
            #if feat.shape[1] > 500:
            #    d_feat = downscale_spectrum(feat.shape[1], 500)
            #else:
            #    d_feat = feat
            ax = plt.subplot2grid( gridsize, (0, col)) # 
            _ = ax.pcolormesh(t1, y, feat, rasterized=True)
            _ = ax.autoscale(tight=True)
            _ = ax.set_title(make_title(mel, log, None, feat))
            _ = ax.set_xlabel("Time (s)")
            _ = ax.set_ylabel("Freq (kHz)") if col == 0 else None
    
            x = np.linspace(0, fs/2./1000, feat.shape[0])
            row = 3
            max_ = np.max(feat, axis=1)
            ax = plt.subplot2grid( gridsize, (row, col)) # 
            _ = ax.plot(x, max_)
            _ = ax.autoscale(tight=True)
            _ = ax.set_title("max [{}]".format(len(max_)))
            _ = ax.set_ylabel("Magnitude") if col == 0 else None
            _ = ax.set_xlabel("Freq (kHz)")
    
            row = 4
            mean_ = np.mean(feat, axis=1)
            std_  = np.std (feat, axis=1)
            ax = plt.subplot2grid( gridsize, (row, col)) # 
            _ = ax.plot(x, mean_)
            _ = ax.plot(x, std_)
            _ = ax.autoscale(tight=True)
            _ = ax.set_title(u"$\mu$, $\sigma$ [{}]".format(len(mean_)))
            _ = ax.set_ylabel("Magnitude") if col == 0 else None
            _ = ax.set_xlabel("Freq (kHz)")
            
    make_suptitle(fig, datum, filename)
    
    return fig
                
def get_data(filename, mel, log, dct, agg, sec_segments=None, reshape=None, verbose=False, modelname="nfc3species"):
    params = "%s%s%s%s%s" % ("mel" if mel else "hertz",
                             "-log" if log else "",
                             "-dct" if dct else "",
                             "-%s" % agg,
                             #"-%s" % sec_segments,
                             "-%.1fsec" % sec_segments if sec_segments else "-entire",
                             )

    picklename = get_output_filename("%s-%s" % (mybasename(filename), params), 
                                     "multiple" + ("%.1f" %sec_segments if sec_segments else ""),
                                               modelname, "npy", removeext=False)
    print picklename
    
    if verbose:
        print u"extracting {}".format(os.path.basename(picklename))
    data = load_from_npy(picklename)
    
    if reshape != None:
        assert len(reshape) == 2, "Reshape must be a 2-element tuple"
        #reshape = (reshape[0], min(reshape[1], 8192))
        try:
            data = data.reshape(*reshape)
        except ValueError as e:
            raise ValueError("Trying to reshape {} into {}: {}".format(data.shape, reshape, e))
    
    return data

def gridplot4(filename, NFFT1, NFFT2, OVERLAP, datum, modelname="nfc3species"):

    def make_title(mel, log, dct, feat):
        if dct == None:
            dct = ""
        elif dct == True:
            dct = "dct"
        else:
            dct = ""
    
        return re.sub(' +', ' ', "%s %s %s %s" % ("mel" if mel else "hertz",
            "log" if log else "",
            dct,
            "[{}x{}]".format(*feat.shape)
            ))        
    ### deprecated. Use gridplot2 instead    


    signal, fs, _ = open_audio(filename)
    #signal = signal[0:5*44100]
        
    mel_spectrum, energy = fbank(signal, samplerate=fs,nfilt=40, 
                     winlen=NFFT1/float(fs), 
                     winstep=(NFFT1-OVERLAP)/float(fs)
                     )
    
    frame_size = NFFT1 #samples
    hop        = NFFT1-OVERLAP #samples
    #hertz_spectrum = half2Darray(np.abs(stft_bysamples(signal, fs, frame_size, hop)))
    hertz_spectrum = stft_bysamples_optimised(signal, fs, frame_size, hop)
    
    fig, axes = plt.subplots(7, 8, figsize=(40,25), dpi=150)
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    

    numcep = 13
    stopfreq_fft2 = float(fs)/NFFT1/2
    START_FREQ = float(fs)/2./1000.*10/(NFFT1/2) ## 22.05*10/32
    
    print "START FREQ:", START_FREQ
    
    
    for mel in [False, True]:
        
        for log in [False, True]:
            
            for dct in [False, True]:
                
                nfeat = None
                row = 0
                col = int(log)+2*int(dct)+4*int(mel)
                
                print "%s %s%s%s" % (os.path.basename(filename), "mel" if mel else "hertz", "-log" if log else "", "-dct" if dct else "")
                
                if mel:
                    feat = mel_spectrum.copy()
                else:
                    feat = hertz_spectrum.copy()
                    
                t  = np.linspace(0, len(signal)/float(fs), len(signal))
                t1 = np.linspace(0, len(signal)/float(fs), feat.shape[0]+1) # time for time-freq spectra
                def f_yaxis(myfeat, start=0, end=fs/2./1000, axis=0):
                    return np.linspace(start, end, myfeat.shape[axis]+1)                 # freq fft1 (0, 22050, 128)
                #f_yaxis = np.linspace(0, fs/2./1000, feat.shape[1]+1)                 # freq fft1 (0, 22050, 128)
                def f_xaxis(myfeat):
                    return np.linspace(0, stopfreq_fft2, myfeat.shape[1]+1)         # freq fft2 (0, 86, 64)
                
                ycep = np.linspace(0, numcep, numcep+1)
                
                x = t1
                y = None    
                
                    
                if log:
                    feat = np.clip(np.log(feat), -100, 100)
                    
                xlab = "Time (s)"
                if dct:
                    

                    y = ycep
                    
                    ylab = "Freq coefficient"
                    
                    mod_shape0 = numcep
                    
                    appendEnergy = False
                    ceplifter = L = 22 # @UnusedVariable
                    
                    dct_ = scipy.fftpack.dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
                    
                    ## lifter
                    nframes,ncoeff = np.shape(dct_) # @UnusedVariable
                    n = np.arange(ncoeff)
                    lift = 1+ (L/2)*np.sin(np.pi*n/L)
                    dct_ = lift*dct_
                    
                    if appendEnergy and mel: dct_[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy                    
                    
                    #dct_ = dct_.T
                    feat = dct_
                else:
                    ylab = "Freq (kHz)"
    
                
                feat = feat.T
                    
                
                d_feat = downscale_spectrum(feat, 500)
                
                if dct:
                    nfeat = numcep
                else:
                    if mel: 
                        nfeat = 40
                        y = np.arange(0, nfeat+1)
                    else:
                        nfeat = 32
                        y = np.linspace(0, fs/2./1000, d_feat.shape[0]+1)
                    
                    
                    
                x = np.linspace(0, len(signal)/float(fs), d_feat.shape[1]+1)
                print "d_feat.shape:", d_feat.shape
                print "x.shape:     ", x.shape 
                print "nfeat", nfeat
                print make_title(mel, log, dct, feat)                

                if not log:
                    d_feat = 10*np.log10(d_feat)

                
                ### spectrum                     
                row = 0
                ax = axes[row, col]
                _ = ax.pcolormesh(x, y, d_feat, rasterized=True)
                _ = ax.autoscale(tight=True)
                _ = ax.set_title(make_title(mel, log, dct, feat))
                _ = ax.set_xlabel(xlab)
                _ = ax.set_ylabel(ylab) if col == 0 else None
                #_ = ax.grid(True)
                    
                ## MOD
                
                downscaled = 50
                ncols = downscaled if downscaled else feat.shape[1]/2
                    
                
                #mod_ = get_data(filename, mel, log, dct, "mod", reshape=(mod_shape0, 64))
                mod_ = 10*np.log10(half2Darray(np.abs(scipy.fft(feat[-nfeat:,], axis=1)), axis=1))
                ### ideally revert to the method below eventually. Unfortunately
                ### if that is rescaled by the featextr then it's not useful here 
                #mod_ = get_data(filename, mel, log, dct, "mod", reshape=(nfeat, ncols), verbose=True, modelname=modelname)
                d_mod_ = downscale_spectrum(mod_, 500)
                
                print "\n\nmod shape:", mod_.shape
                print "feat shape:", feat.shape
                print "d_mod_ shape:", d_mod_.shape, "\n\n" 
                
                row += 1
                ax = axes[row, col]
                _ = ax.pcolormesh(f_xaxis(d_mod_), f_yaxis(d_mod_, start=START_FREQ), 10*np.log10(d_mod_), rasterized=True)
                _ = ax.autoscale(tight=True)            
                _ = ax.set_title("MOD [{}x{}] [{}xd{}]".format(*(mod_.shape+d_mod_.shape)))
                _ = ax.set_xlabel("Modulation freq (Hz)")
                _ = ax.set_ylabel("Freq (kHz)") if col == 0 else None
                #_ = ax.grid(True)
                
                
                ## MOD 10
                
                mod10 = mod_[:,0:mod_.shape[1]/10]
                d_mod10 = downscale_spectrum(mod10, 500)
                x = np.linspace(0, stopfreq_fft2/10, d_mod10.shape[1]+1)
                
                row += 1
                ax = axes[row, col]
                _ = ax.pcolormesh(x, f_yaxis(d_mod10, start=START_FREQ), 10*np.log10(d_mod10), rasterized=True)
                _ = ax.autoscale(tight=True)            
                _ = ax.set_title("MOD10 [{}x{}] [{}xd{}]".format(*(mod10.shape+d_mod10.shape)))
                _ = ax.set_xlabel("Modulation freq (Hz)")
                _ = ax.set_ylabel("Freq (kHz)") if col == 0 else None
                #_ = ax.grid(True)                

                
                nbins=48
                mod_logged = log_mod(mod_.T, fs, NFFT1, nbins=nbins).T
                ## LOG MOD
                row += 1
                ax = axes[row, col]
                x = np.linspace(0, nbins, nbins+1)
                _ = ax.pcolormesh(x, f_yaxis(mod_logged, start=START_FREQ), 10*np.log10(mod_logged), rasterized=True)
                _ = ax.autoscale(tight=True)            
                _ = ax.set_title("LOG MOD [{}x{}]".format(*mod_logged.shape))
                _ = ax.set_xlabel("Modulation coefficient")
                _ = ax.set_ylabel("Freq (kHz)") if col == 0 else None
                #_ = ax.grid(True)
                
                
                row += 1
                #mod0 = get_data(filename, mel, log, dct, "mod0", verbose=True, modelname=modelname)
                mod0 = mod_[:,0]                
                x = np.linspace(START_FREQ, fs/2./1000, len(mod0))
                ax = axes[row, col] 
                _ = ax.plot(x, mod0)
                _ = ax.autoscale(tight=True, axis="x")
                _ = ax.set_title("MOD[0] [{}]".format(len(mod0)))
                _ = ax.set_ylabel("Magnitude") if col == 0 else None
                _ = ax.set_xlabel("Freq (kHz)")
                
                
                row += 1
                meanstd = get_data(filename, mel, log, dct, u"μ+σ", modelname=modelname)
                assert len(meanstd) == 2*nfeat, "len(meanstd): %s, 2*nfeat: %s" % (str(len(meanstd)), str(2*nfeat))
                mean_ = meanstd[nfeat:]
                std_  = meanstd[:nfeat]
                print "\n\nx shape:", x.shape
                print "mean_shape", mean_.shape, "\n\n"
                ax = axes[row, col]  
                _ = ax.plot(x, mean_, label="$\mu$")
                _ = ax.plot(x, std_, label="$\sigma$")
                _ = ax.autoscale(tight=True, axis="x")
                _ = ax.set_title(u"$\mu$, $\sigma$ [{}]".format(len(mean_)))
                _ = ax.set_ylabel("Magnitude") if col == 0 else None
                _ = ax.set_xlabel("Freq (kHz)")
                _ = ax.legend(loc="upper left") if col == 0 else None

                row += 1                
                max_ = get_data(filename, mel, log, dct, "max", modelname=modelname)
                ax = axes[row, col] 
                _ = ax.plot(x, max_)
                _ = ax.autoscale(tight=True, axis="x")
                _ = ax.set_title("max [{}]".format(len(max_)))
                _ = ax.set_ylabel("Magnitude") if col == 0 else None
                _ = ax.set_xlabel("Freq (kHz)")
        
                
            
    #make_suptitle(fig, datum, filename)
    
    return fig
    

def pdfpages_savefig(filename):
    pdf.savefig()  # @UndefinedVariable
    
def plt_savefig(filename):
    plt.savefig(filename)
    
def close():
    plt.close()