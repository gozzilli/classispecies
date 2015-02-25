import os

modelname  = "default"
classifier = "decisiontree"
analyser   = "mfcc"

## settings
SPECTR_PLOT     = False
CONFUSION_PLOT  = False
FEATURES_PLOT   = True   # feature matrix of all files combined (one vector per file or chunk)
IMPORTANCE_PLOT = False  # most important features according to classifier
TREE_PLOT       = False
CLASSIF_PLOT    = False
FEATURE_ONEFILE_PLOT = False # feature matrix of one file, before being compress to a vector

def do_plot():
    print "AAAAAAAAAAAAAAAAA"
    print SPECTR_PLOT, CONFUSION_PLOT, FEATURES_PLOT, IMPORTANCE_PLOT, TREE_PLOT, FEATURE_ONEFILE_PLOT
    print "BBBBBBBBBBBBBBBBB"
    return any((SPECTR_PLOT, CONFUSION_PLOT, FEATURES_PLOT, IMPORTANCE_PLOT, TREE_PLOT, FEATURE_ONEFILE_PLOT))

IS_MPL_BACKEND_SET = False
def is_mpl_backend_set():
    return IS_MPL_BACKEND_SET

def set_mpl_backend():
    import matplotlib
    matplotlib.use(MPL_BACKEND)
    
savefig         = True
show            = False

N_MEL_FILTERS   = 40
highpass_cutoff = 7000 ## set to 0 to disable high-pass filter
mod_cutoff      = 50
delta           = 0
test_size       = 0.5
normalise       = True
n_segments      = None
sec_segments    = None
extract_mean    = False
extract_std     = False
extract_max     = False
whiten_feature          = False
whiten_features         = False
whiten_feature_matrix   = False

extract_mel     = False
extract_dolog   = False
extract_dct     = False
extract_fft2    = False
extract_logmod  = False
agg             = None
downscale_factor = None

ceplifter       = 22
NFFT1           = 64
NFFT2           = NFFT1/2
OVERLAP         = 0
MOD_TAKE1BIN    = False

logger_level   = "INFO"


SPLIT_TRAINING_SET  = False
SPLIT_METHOD        = "split-after"
FORCE_FEATXTR       = False
FORCE_FEATXTRALL    = False
FORCE_MULTIRUNNER_RECOMPUTE = False
MULTILABEL          = True
MODEL_BINARY        = False
MULTICORE           = False
DUMP_REPORT         = True
SAVE_TO_DB          = False
RANDOM_SEED         = 12345



MPL_FORMAT      = "svg"
MPL_BACKEND     = "agg"


_here = os.path.dirname(os.path.abspath(__file__))
#SOUNDPATHS = {"default" : 
#                { 'train' : '',
#                  'test'  : '',
#                }
#              }
LABELS = {
    'default' :
        { 'train' : '',
          'test'  : '',
        },
    }
OUTPUT_DIR = 'outputs2' #'outputs'

CLASSES         = [ 
'chl',
'fri',
'bla',
'syl',
'car',
]
NCLASSES        = len(CLASSES)
NMFCCS          = 13 # number of MFCCs.
NFEATURES       = NMFCCS * 3


from types import ModuleType, FunctionType, BuiltinFunctionType, TypeType
from pprint import pformat
from md5 import md5
from collections import OrderedDict
import numpy as np

def get_all(prettify=True, sort=False, as_string=True):
    sett = OrderedDict()
    gl = globals()
    for k,v in gl.iteritems():
        #print k,v
        if not k.startswith("__") and not isinstance(v, (ModuleType, FunctionType, BuiltinFunctionType, TypeType)):
            if isinstance(v, np.ndarray):
                if len(v.shape) == 1:
                    v = list(v)
                else:
                    raise Exception("arghh")
            sett[k] = v
            
    if sort:
        sett = sorted(sett)
        
    if as_string:
        sett = "\n".join(["%s = %s" % (k,pformat(v) if prettify else v) for k,v in sett.iteritems()])

    return sett
    
def digest():
    return md5(get_all(prettify=False, sort=True)).hexdigest()
