DESCRIPTION = '''
Launch classispecies.

All parameters can be specified in the settings file or on command line.
Command line takes precedence over settings file.

Params:

* path to the sound files to classify 
* analysis method (features extraction)
* classification method
    
'''

import argparse
from classispecies import settings

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--classifier', '-c', type=str,
                   help='classification method')
parser.add_argument('--analyser', '-a', type=str,
                   help='feature extraction method')

args = parser.parse_args()

settings.analyser = args.analyser
settings.classifier = args.classifier

print args

