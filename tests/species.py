import sys, os
import re
import gc
from os.path import expanduser as home
from tempfile import mkdtemp

import subprocess
import shutil

import pandas as pd
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.interpolate import interp1d

from features import logfbank, fbank
import wavio

from cs.plot import gridplot2, gridplot3, gridplot4, plt_savefig, pdfpages_savefig, close

from classispecies.utils.signal import open_audio, half2Darray, stft_bysamples


UKORTH_DIR = home("~/Dropbox/Shared/Orthoptera Sound App/species_recordings")
BL_DIR = home("~/cicada-largefiles/bl-files/all-16bit/")
BL_DATA = home("~/Dropbox/Uni/DTC1/Insects/BL/datasets/dataset4.xlsx")
PLOT_METHOD = 3

TEST_FILE = '8kHz wave with 6Hz chirp.wav'

nfc = pd.read_csv('../cicada-train.csv',usecols=(0,), header=None)
cic = np.ravel(nfc[nfc[0].str.contains("_cicada")].as_matrix())
rbc = np.ravel(nfc[nfc[0].str.contains("_roesel")].as_matrix())
dbc = np.ravel(nfc[nfc[0].str.contains("_dark")].as_matrix())
NFC_DATA = np.concatenate( (cic, rbc, dbc) )
 

def allplot(wavdir, outfilename, savefig_, NFFT1, NFFT2, OVERLAP, counter=0, total=0, data=None, limit=None, modelname="nfc3species"):
    
    allsoundfiles = sorted(os.listdir(wavdir)) if isinstance(wavdir, str) else wavdir
    #for soundfile in ['8kHz wave with 6Hz chirp.wav'] + sorted(os.listdir(wavdir)):
    for soundfile in allsoundfiles[0:limit]:

        counter += 1
        if not soundfile.lower().endswith(".wav"): continue

        sys.stdout.write("\r[%d/%d] %s" %(counter, total, outfilename % counter))
        sys.stdout.flush()

        try:
            
            if not TEST_FILE == soundfile:
                soundfile = os.path.join(wavdir, soundfile)
                
            signal, fs, enc = open_audio(soundfile)
            
            datum = None    
            if isinstance(data, pd.DataFrame):
                datum=data[data.sound.str.lower() == soundfile[0:-4]].squeeze()
                if datum.empty:
                    datum = None
            
            if PLOT_METHOD == 2:        

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
                hertz_feat = half2Darray(abs(stft_bysamples(signal, fs, frame_size, hop))).T
                hertz_fft  = half2Darray(np.abs(scipy.fft(hertz_feat, n=NFFT2)))
    
                gridplot2(soundfile, signal, fs, melfft, hertz_fft, NFFT1, NFFT2, datum)
            
                
            
            if PLOT_METHOD == 3:
                fig = gridplot4(soundfile, NFFT1, NFFT2, OVERLAP, datum, modelname=modelname )
            
            else:
                raise NotImplemented
            
            savefig_(outfilename % counter)  # saves the current figure into a pdf page
            close()
            
            
        except MemoryError:
            print "\nmemory error on %s" % soundfile
            mel_feat = melfft = hertz_feat = hertz_fft = None
            close() 
            raise
        
        

    
WAVDIR_OPTS = [('outputs/nfc-%02d-all.pdf', NFC_DATA, None, "nfc3species"),
               #('outputs/uk-orthoptera_2xFFT-%02d-all.pdf', UKORTH_DIR, None, "ukorthoptera"), 
               #('outputs/bl-orthoptera_2xFFT-%02d-all.pdf', BL_DIR, BL_DATA, "blorthoptera")
               ]
POWER_OPTS  = [8]#[8, 9]
LIMIT = None

counter = 0
#total = sum([len(os.listdir(wavdir)) for _, wavdir, _ in WAVDIR_OPTS]) * len(POWER_OPTS)
total = len(NFC_DATA[0:LIMIT]) + len(os.listdir(UKORTH_DIR)[0:LIMIT])

for pdffilename, wavdir, datafile, modelname in WAVDIR_OPTS:
    for nfft_power in POWER_OPTS:

        try:
            NFFT1 = 64
            OVERLAP = 0
            NFFT2 = NFFT1/2
        
        
            tempdir = mkdtemp()
            print tempdir
            outfilename = os.path.join(tempdir, "temp_%04d.pdf")
            allfilename = pdffilename % NFFT1
            
            if datafile:
                data = pd.read_excel(datafile)
            else:
                data = None
        
        
            allplot(wavdir, outfilename, plt_savefig, NFFT1, NFFT2, OVERLAP, counter=counter, total=total, data=data, limit=LIMIT, modelname=modelname)
        
        finally:
            
            try:
                reducedfilename = allfilename[0:-4]+"-reduced.pdf"
                #!pdftk $tempdir/*.pdf cat output $allfilename
                _exit_code = subprocess.call("pdftk %s/*.pdf cat output %s" %(tempdir, allfilename), shell=True) 
                if _exit_code == 0:
                    print "Output in", allfilename 
                    _exit_code = subprocess.call("gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=%s %s" % (reducedfilename, allfilename), shell=True)
                    if _exit_code == 0:
                        print "Reduced output in", reducedfilename 
                        shutil.rmtree(tempdir)
                else:
                    print "PDFTK failed"
                
            except KeyboardInterrupt:
                shutil.rmtree(tempdir)
                raise
     