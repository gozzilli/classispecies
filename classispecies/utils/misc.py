# -*- coding: utf-8 -*-

from __future__ import print_function
import warnings, functools
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

from classispecies import settings
if not settings.is_mpl_backend_set():
    settings.set_mpl_backend()

from matplotlib import pyplot as plt
from IPython.display import FileLink, display

import random
import string

from classispecies import settings

is_logging_configured = False


def mybasename(path):
    ''' return file name with no directory and no extension '''
    return os.path.splitext(os.path.basename(path))[0]

def get_output_filename(pdffilename, component, modelname, ext, removeext=True):
    return os.path.join( settings.OUTPUT_DIR, modelname, component, 
                         "%s.%s" % (mybasename(pdffilename) if removeext else pdffilename, ext))

def make_output_filename(pdffilename, component, modelname, ext, removeext=True):
    outname = get_output_filename(pdffilename, component, modelname, ext, removeext)
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

def get_soundpath(soundpaths, set_="train"):
    return get_path(soundpaths, set_)

def dump_report(variables, modelname):
    
    sett = ""
    vars_ = OrderedDict(sorted(vars(settings).items()))

    for k,v in vars_.iteritems():
        if not k.startswith("__") and not isinstance(v, (ModuleType, FunctionType, BuiltinFunctionType, TypeType)):
            sett += "%s = %s\n" % (k, pformat(v))
    
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter  # @UnresolvedImport
    
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
    template = env.get_template('serve/templates/simulationrun.html')
    
    def copy_diagram(pdffilename, component, varname):
    
        plot = get_output_filename(pdffilename, component, modelname, "svg")
        if os.path.exists(plot):
            plot_report = make_output_filename(pdffilename+"_"+filetimestamp,
                                               modelname, "simulationrun", "svg")
            shutil.copyfile(plot, plot_report)
            variables[varname] = os.path.basename(plot_report)

    #if settings.MULTILABEL:
    
    copy_diagram("classifplot", "classify", "output_plot")
    if settings.FEATURES_PLOT:
        copy_diagram("features", "featxtr", "features_plot")
    
    with open(outfilename, 'wb') as f:
        f.write(template.render(variables))
        
    print ("dumping report to:")
    display(FileLink(outfilename.replace(" ", "\\ ")))


def dump_to_pickle(obj, outfilename):
    with open(outfilename, 'wb') as outfile:
        pickle.dump( obj, outfile )

def dump_to_npy(obj, outfilename):
    np.save(outfilename, obj)

def load_from_pickle(picklename):
    with open(picklename, 'rb' ) as infile:
        obj = pickle.load( infile )
    return obj

def load_from_npy(picklename):
    return np.load(picklename)

def plot_or_show(fig, pdffilename, tight=True, **kwargs):
    if tight:
        fig.tight_layout()
    if settings.savefig:
        print ("saving", pdffilename)
        fig.savefig(pdffilename, **kwargs)
    if settings.show:
        print ("showing")
        fig.show()
    #fig.close()
    fig.clf()
    plt.close('all')
    
    print ("done")

def config_logging(name):
    
    global is_logging_configured

    if not is_logging_configured:
        if not os.path.exists('outputs/logs'):
            os.makedirs('outputs/logs')
        with open( os.path.join(os.path.dirname(__file__),'logging.yaml'), 'r') as f:
            logging.config.dictConfig(yaml.load(f))
        
        is_logging_configured = True

    if name in ['result_to_file', 'result_to_grid', 'f1', 'confusion']:
        temp_logger = logging.getLogger(name)
        
    else:
        temp_logger = logging.getLogger("classispecies")
        temp_logger.name = name
        temp_logger.setLevel(settings.logger_level)
        # temp_logger.debug("temp_logger ready")
    return temp_logger

def logger_shutdown():
    logging.shutdown()


def rprint(str_):
    #if settings.logger_level == "DEBUG":
        print (("\r%s" % str_)[:80], end="")
        sys.stdout.flush()
    

def savedata_for_azure(pdffilename, obj):
    ''' export feature data to file so that they can be processed by Azure ML '''

    np.savetxt(make_output_filename(pdffilename, "", settings.modelname, "csv"), obj, delimiter=",",
               fmt="%s", comments = '', header= ",".join(["label"] +
                                #["mean%d" % x for x in range(settings.NMFCCS-1)] +
                                #["std%d"  % x for x in range(settings.NMFCCS-1)] +
                                ["max%d"  % x for x in range(settings.NMFCCS-1)] ))
    
def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1, 
            module=func.__module__
        )
        return func(*args, **kwargs)
    return new_func

def key_gen(size=5, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def get_an():
    return "%s %s%s%s%s%s" % (settings.classifier, 
         "mel" if settings.extract_mel else "hertz",
         " log" if settings.extract_dolog else "",
         " dct" if settings.extract_dct else "",
         " %s" % settings.agg,
         " %.1fsec" % settings.sec_segments if settings.sec_segments else " entire")
    
def get_agg():
    ''' Get the name of the aggregator '''
    agg = ""
    if settings.extract_max:
        agg += "max"
    if settings.extract_mean:
        agg += "μ"
    if settings.extract_std:
        agg += "σ"
    if settings.extract_fft2:
        agg = "mod"
    if settings.extract_logmod:
        agg = "logmod"
    return agg