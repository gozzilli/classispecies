'''
Created on 28 Jan 2015

@author: Davide Zilli
'''



import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(stream=sys.stderr)


print "running cs"
from serve import app, cs
app.debug = True
#app.register_blueprint(cs, url_prefix='/cs')
app.register_blueprint(cs)

application = app
