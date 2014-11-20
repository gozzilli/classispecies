# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:49:28 2014

@author: Davide Zilli
"""

from classispecies.utils import misc
from classispecies import settings

if settings.PLOT:
    import matplotlib
    matplotlib.use(settings.MPL_BACKEND)
    from matplotlib import pyplot as plt, cm


def classif_plot(self, labels_testing, prediction):
            
    print ("plotting classification plot")
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(131)
    ax.autoscale(tight=True)
    ax.pcolormesh(labels_testing, rasterized=True)
    ax.set_title("ground truth")
    
    ax = fig.add_subplot(132)
    ax.autoscale(tight=True)
    ax.pcolormesh(self.prediction, rasterized=True)
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
    misc.plot_or_show(fig, filename=outfilename)

def feature_plot(data_training, modelname, format_):   
    fig = plt.figure()
    plt.pcolormesh(data_training, rasterized=True)
    plt.autoscale(tight=True)
    plt.xlabel("feature")
    plt.ylabel("sound file")
    outfilename = misc.make_output_filename("features", "featxtr", modelname, format_)

    misc.plot_or_show(fig, filename=outfilename)