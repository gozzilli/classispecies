# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:49:28 2014

@author: Davide Zilli
"""

import os
from classispecies.utils import misc
from classispecies import settings
from classispecies.utils.signal import downscale_spectrum
import numpy as np

if not settings.is_mpl_backend_set():
    settings.set_mpl_backend()

from matplotlib import pyplot as plt, cm


def classif_plot(labels_testing, prediction):
            
    print ("plotting classification plot")
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(131)
    ax.autoscale(tight=True)
    ax.pcolormesh(labels_testing, rasterized=True)
    ax.set_title("ground truth")
    
    ax = fig.add_subplot(132)
    ax.autoscale(tight=True)
    ax.pcolormesh(prediction, rasterized=True)
    ax.set_title("prediction")
    
    ax = fig.add_subplot(133)
    
    img = ax.pcolormesh(labels_testing - prediction, cmap=cm.get_cmap('PiYG',3), rasterized=True)
    cb = plt.colorbar(img, ax=ax)
    cb.set_ticks([-1,0,1], False)
    cb.set_ticklabels(["FP", "T", "FN"], False)
    cb.update_ticks()
    ax.autoscale(tight=True)
    
    print ("saving classification plot")

    # fig.savefig(misc.make_output_filename("classifplot", "classify", settings.modelname, settings.MPL_FORMAT), dpi=150)
    outfilename = misc.make_output_filename("classifplot", "classify", settings.modelname, settings.MPL_FORMAT)
    misc.plot_or_show(fig, pdffilename=outfilename)

def feature_plot(data_training, modelname, format_):
    
    fig = plt.figure()
    if data_training.shape[1] > 2000:
        d = downscale_spectrum(data_training, 1000)
    else: 
        d = data_training
        
    #print "feature plot: %s (plotted at %s)" % (data_training.shape, d.shape)
    plt.pcolormesh(d, rasterized=True)
    plt.autoscale(tight=True)
    plt.xlabel("feature")
    plt.ylabel("sound file")
    outfilename = misc.make_output_filename("features", "featxtr", modelname, format_)

    misc.plot_or_show(fig, pdffilename=outfilename)
    
def file_plot(signal, rate, feat, soundfile):
    feat = feat.copy()
    fig = plt.figure( figsize=(10,4) )
    plt.subplot(121)
    plt.specgram(signal, Fs=rate, NFFT=settings.NFFT1, noverlap=0, rasterized=True)
    plt.autoscale(tight=True)
    plt.xlabel("Time (sec)")
    plt.ylabel("Freq (Hz)")
    plt.suptitle("%s :: %s" % (misc.get_an(), os.path.basename(soundfile)), fontsize=8)
    
    feat = downscale_spectrum(downscale_spectrum(feat, 500, axis=1), 50, axis=0)
    plt.subplot(122)
    plt.pcolormesh(10*np.log10(feat), rasterized=True)
    plt.autoscale(tight=True)
    
    outfilename = misc.make_output_filename("features-onefile-%d" %os.getpid(), "featxtr", settings.modelname, "pdf")#settings.MPL_FORMAT)

    misc.plot_or_show(fig, pdffilename=outfilename)
