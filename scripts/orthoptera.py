import os
from classispecies import settings
from classispecies.utils import misc

settings.modelname  = "orthoptera"
settings.classifier = "decisiontree"
settings.analyser   = "mel-filterbank"

settings.SOUNDPATHS.update({"Boa" : os.path.expanduser('~/Dropbox/Uni/DTC1/projects/cicada-detect/src/biomon/sound/DutchOrthoptera/orthoptera'),
                            "default" : os.path.expanduser('~/Dropbox/Uni/DTC1/projects/cicada-detect/src/biomon/sound/DutchOrthoptera/orthoptera'),
                            })

settings.CLASSES = sorted(set([x[0:3] for x in os.listdir(misc.get_soundpath())]))
settings.NCLASSES = len(settings.CLASSES)

from classispecies import classispecies
classispecies.run()