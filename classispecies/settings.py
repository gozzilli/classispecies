import os, sys, socket

modelname  = "default"
classifier = "decisiontree"
analyser   = "mfcc"

## settings
SPECTR_PLOT     = False
CONFUSION_PLOT  = True
FEATURES_PLOT   = True
IMPORTANCE_PLOT = True # most important features according to classifier
TREE_PLOT       = False
CLASSIF_PLOT    = True
PLOT            = any((SPECTR_PLOT, CONFUSION_PLOT, FEATURES_PLOT, IMPORTANCE_PLOT, TREE_PLOT))

savefig         = True
show            = False

N_MEL_FILTERS   = 40
highpass_cutoff = 500 ## set to 0 to disable high-pass filter
delta           = 0
normalise       = False
n_segments      = None
sec_segments    = None
extract_mean    = True
extract_std     = False
extract_max     = False
whiten_feature          = False
whiten_features         = False
whiten_feature_matrix   = False


SPLIT_TRAINING_SET  = False
FORCE_FEATXTR       = False
MULTILABEL          = True
MULTICORE           = False


MPL_FORMAT      = "svg"
MPL_BACKEND     = "agg"
if PLOT:
    import matplotlib
    matplotlib.use(MPL_BACKEND)

_here = os.path.dirname(os.path.abspath(__file__))
SOUNDPATHS = {"default"        : os.path.join(_here, "../../Resources/birds/")}
LABELS = {
    'default' :
        { 'train' : '',
          'test'  : '',
        },
    }
OUTPUT_DIR = 'outputs'

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


from types import ModuleType, FunctionType
from pprint import pformat
from md5 import md5

def get_all(prettify=True, sort=False):
    sett = []
    gl = globals()
    for i in gl:
        if not i.startswith("__") and not isinstance(gl[i], ModuleType) and not isinstance(gl[i], FunctionType):
            sett.append( "%s = %s" % (i, pformat(gl[i]) if prettify else gl[i]))
    if sort:
        return "\n".join(sorted(sett))
    else:
        return "\n".join(sett)
    
def digest():
    return md5(get_all(prettify=False, sort=True)).hexdigest()
