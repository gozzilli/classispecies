$$$$$$$$$$$$$$$$$$$$$$$
SoundTrap Classispecies
$$$$$$$$$$$$$$$$$$$$$$$

Bioacoustic classifier written in Python

Features
========

Full classification pipeline from audio file to output class:

* Read audio files
* Analyse them, extracting a range of different features
* Train a classification model
* Evaluate the classification against test data
* Assess the classification output with different methods
* Output most likely class for each recording and relative probabilities
* Output matrix of all combination of parameters considered
  
Engineered to be extended:

* Plug-in your own feature extraction method
* Plug-in your own classification method
* Serialisation of intermediate steps, to avoid recomputing identical steps  
* Object-oriented design
* Separation of model and controller
* Separation of settings configuration to customise all parameters in one place.

Engineered for performace:

* Multi-core operation that can process recordings, train and validate results 
  in parallel (through the ``multiprocessing`` library)
* Dynamic comparison of different parameters (also saved to database and served 
  in HTML)
* Logging of model operation with different levels of verbosity

Engineered for collaboration:

* Utilities for Comma-separated values (CSV) formatting of input data
* Utilities for conversion of extracted features to 
  `ARFF <http://www.cs.waikato.ac.nz/ml/weka/arff.html>`_
  (`Weka <http://www.cs.waikato.ac.nz/ml/weka/>`_ format).

Focus on results:

* Store results in a mongo database
* Serve results as a web page (HTML over configurable HTTP server)


Usage
=====

This program classifies wildlife species according to a given model. A model is
made of:

1.  A feature extraction method
2.  A classifier
3.  A data set of sound files
4.  A set of parameters [optional].

We provide a set of example models that can be copied for a quick start:

*   ``orthoptera.py``    -- all dutch orthoptera
*   ``birds.py``         -- a selection of 5 species of birds
*   ``nfc-vs-dbc.py``    -- New Forest cicada and Roesel bush cricket.
*   ``nips4b.py``        -- NIPS classification challenge
*   ``ukorthoptera.py``  -- set of 28 orthoptera species in the UK. 

``orthoptera.py``
   all dutch orthoptera
``birds.py``
   a selection of 5 species of birds
``nfc-vs-dbc.py``
   New Forest cicada and Roesel bush cricket.
``nips4b.py``
   NIPS classification challenge
``ukorthoptera.py``
   set of 28 orthoptera species in the UK. 


Existing features extraction methods are:

*   ``mfcc``            -- Mel Frequency Cepstral Coefficients
*   ``mel-spectrum``    -- Mel Spectrum
*   ``hertz-spectrum``  -- Hertz Spectrum
*   ``sp-kmeans``       -- Spherical k-means clustering

Existing supervised and unsupervised classifiers are:

*   ``decisiontree``    -- Decision Tree Classifier
*   ``randomforest``    -- Random Forest (Ensamble of Decision Trees) 
*   ``kmeans``          -- K-means clustering


