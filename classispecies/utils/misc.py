# -*- coding: utf-8 -*-

from __future__ import print_function
import os, socket, sys
import pickle
import shutil
from collections import OrderedDict

import logging.config
import yaml
import numpy as np

from datetime import datetime
from pprint import pformat
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
from types import ModuleType, FunctionType, BuiltinFunctionType, TypeType

from matplotlib import pyplot as plt
from IPython.display import FileLink, display

from classispecies import settings


def mybasename(path):
    ''' return file name with no directory and no extension '''
    return os.path.splitext(os.path.basename(path))[0]

def get_output_filename(filename, component, modelname, ext, removeext=True):
    return os.path.join( settings.OUTPUT_DIR, modelname, component, 
                         "%s.%s" % (mybasename(filename) if removeext else filename, ext))

def make_output_filename(filename, component, modelname, ext, removeext=True):
    outname = get_output_filename(filename, component, modelname, ext, removeext)
    outdir  = os.path.dirname(outname)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outname

def gethostname():
    return socket.gethostname()

def get_path(pathvar, set_):

    hostpaths = pathvar.get(gethostname(), pathvar.get("default"))

    if isinstance(hostpaths, dict):
        return hostpaths[set_]
    else:
        return hostpaths

def get_soundpath(soundpaths=settings.SOUNDPATHS, set_="train"):
    return get_path(soundpaths, set_)

def dump_report(variables, modelname):
    
    sett = ""
    vars_ = OrderedDict(sorted(vars(settings).items()))

    for k,v in vars_.iteritems():
        if not k.startswith("__") and not isinstance(v, ModuleType) \
                                  and not isinstance(v, FunctionType) \
                                  and not isinstance(v, BuiltinFunctionType) \
                                  and not isinstance(v, TypeType):
            sett += "%s = %s\n" % (k, pformat(v))
    
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    
    lexer = get_lexer_by_name("python", stripall=True)
    formatter = HtmlFormatter(linenos=True, cssclass="source")
    result = highlight(sett, lexer, formatter)
    
    variables["settings"] = result
    variables["modelname"] = modelname
    
    now = datetime.now()
    filetimestamp = now.strftime("%Y_%m_%d__%H:%M:%S")
    variables["now"] = unicode(now.strftime("%d %b %Y, %H:%M:%S"))

    outfilename = make_output_filename(now.strftime("%Y_%m_%d__%H:%M:%S"),
                                       modelname, "simulationrun", "html")
    
    env = Environment(loader=FileSystemLoader('.', encoding='utf-8'))
    template = env.get_template('simulationrun.template')
    
    def copy_diagram(filename, component, varname):
    
        plot = get_output_filename(filename, component, modelname, "svg")
        if os.path.exists(plot):
            plot_report = make_output_filename(filename+"_"+filetimestamp,
                                               modelname, "simulationrun", "svg")
            shutil.copyfile(plot, plot_report)
            variables[varname] = os.path.basename(plot_report)

    #if settings.MULTILABEL:
    copy_diagram("classifplot", "classify", "output_plot")
    copy_diagram("features", "featxtr", "features_plot")
    
    with open(outfilename, 'wb') as f:
        f.write(template.render(variables))
        
    print ("dumping report to:")
    display(FileLink(outfilename.replace(" ", "\\ ")))


def dump_to_pickle(obj, outfilename):
    with open(outfilename, 'wb') as outfile:
        pickle.dump( obj, outfile )

def load_from_pickle(picklename):
    with open(picklename, 'rb' ) as infile:
        obj = pickle.load( infile )
    return obj

def plot_or_show(fig, filename=None, tight=True, **kwargs):
    if tight:
        fig.tight_layout()
    if settings.savefig:
        print ("saving", filename)
        fig.savefig(filename or settings.filename, **kwargs)
    if settings.show:
        print ("showing")
        fig.show()
    #fig.close()
    fig.clf()
    plt.close('all')
    
    print ("done")

def config_logging(name):
    with open( 'logging.yaml', 'r') as f:
        logging.config.dictConfig(yaml.load(f))

    if name in ['result_to_file', 'result_to_grid']:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger("classispecies")
        logger.name = name
        logger.setLevel(settings.logger_level)
        # logger.debug("logger ready")
    return logger

def logger_shutdown():
    logging.shutdown()


def rprint(str_):
    print ("\r%s\r" % str_, end="")
    sys.stdout.flush()
    

def savedata_for_azure(filename, obj):
    ''' export feature data to file so that they can be processed by Azure ML '''

    np.savetxt(make_output_filename(filename, "", settings.modelname, "csv"), obj, delimiter=",",
               fmt="%s", comments = '', header= ",".join(["label"] +
                                #["mean%d" % x for x in range(settings.NMFCCS-1)] +
                                #["std%d"  % x for x in range(settings.NMFCCS-1)] +
                                ["max%d"  % x for x in range(settings.NMFCCS-1)] ))