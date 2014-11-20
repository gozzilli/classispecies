import os
from classispecies import settings

settings.modelname  = "birds"
settings.classifier = "decisiontree"
settings.analyser   = "mfcc"

_here = os.path.dirname(os.path.abspath(__file__))
settings.SOUNDPATHS.update({"MSRC-3617038"   : "C:/Users/t-davizi/OneDrive/Work/SoundTrap/Resources/birds/xenocanto",
              "DavsBook.local" : os.path.join(_here, "../Resources/birds/"),
              "davsbook"       : os.path.join(_here, "../Resources/birds/"),
              "Boa"            : os.path.join(_here, "../Resources/birds/"),
              "default"        : os.path.join(_here, "../Resources/birds/")})

settings.CLASSES         = [ 
'chl',
'fri',
'bla',
'syl',
'car',
]
settings.NCLASSES = len(settings.CLASSES)

from classispecies import classispecies
classispecies.run()
