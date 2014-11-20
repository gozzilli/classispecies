$$$$$$$$$$$$$$$$$$$$$$$$$$$
  SoundTrap Classispecies
$$$$$$$$$$$$$$$$$$$$$$$$$$$

Usage
=====

This program classifies wildlife species according to a given model. A model is
made of:

1.  a feature extraction method
2.  a classifier
3.  a data set of sound files
4.  an optional set of parameters.

Existing models are:

*   orthoptera.py -- all dutch orthoptera
*   birds.py      -- a selection of 5 species of birds
*   nfc-vs-dbc.py -- New Forest cicada and Roesel bush cricket.

Existing features extraction methods are:

*   mfcc            -- Mel Frequency Cepstral Coefficients
*   mel-spectrum    -- Mel Spectrum
*   hertz-spectrum  -- Hertz Spectrum
*   sp-kmeans       -- Spherical k-means clustering

Existing (un)supervised classifiers are:

*   decisiontree    -- Decision Tree Classifier
*   kmeans          -- K-means clustering


Deprecated files
================

classispecies.py
justplot_classispecies.py
azure_classify.py

birds/classispecies.py
birds/confusion.py

utils/confusion.py
