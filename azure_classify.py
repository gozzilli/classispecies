import numpy as np

import scipy.io.wavfile as wav

from features import mfcc

import json
import urllib2
from termcolor import colored

NMFCCS = 13


URL     = 'https://ussouthcentral.services.azureml.net/workspaces/f42d8692670644afb5f08c938bf50abb/services/38a6a1ba80cb4f73b9f51acbc10670a2/score'
API_KEY = 'ncSK4Xl68XBd8F/ey1el+h7GDisuLxSWj2y0zFXQeJ+EPs4aKsOHBpFP7+iTNCcBPX1TGbN8Txqap3ei9y44CQ==' 
HEADERS = {'Content-Type':'application/json', 'Authorization':('Bearer '+ API_KEY)}

data =  {
            "Id": "score00001",
            "Instance": {
                "FeatureVector": {
                },
                "GlobalParameters": {
                }
            }
        }

def apply_workaround_bug11220():
    '''
    A bug in python2 raises an SSL error if SSL v23 is used. This workaround
    solves the problem. 
    '''
    
    # BEGIN workaround for bug 11220 (http://bugs.python.org/issue11220)
    import httplib, ssl, socket
    class HTTPSConnectionV3(httplib.HTTPSConnection):
        def __init__(self, *args, **kwargs):
            httplib.HTTPSConnection.__init__(self, *args, **kwargs)
            
        def connect(self):
            sock = socket.create_connection((self.host, self.port), self.timeout)
            if self._tunnel_host:
                self.sock = sock
                self._tunnel()
            try:
                self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, ssl_version=ssl.PROTOCOL_SSLv3)
            except ssl.SSLError:
                print("Trying SSLv3.")
                self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, ssl_version=ssl.PROTOCOL_SSLv23)
                
    class HTTPSHandlerV3(urllib2.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(HTTPSConnectionV3, req)
    # install opener
    urllib2.install_opener(urllib2.build_opener(HTTPSHandlerV3()))
    ### END workaround

def query():
    
    body = str.encode(json.dumps(data))
    
    req = urllib2.Request(URL, body, HEADERS) 
    response = urllib2.urlopen(req)
    result = json.loads(response.read())
    #print(result)

    print "\ngiven: %s, classified as: %s" % (colored(result[0], "red"), colored(result[-2], "red"))

    success = "correct!" if result[0] == result[-2] else "wrong :("
    print colored("%s (%s)" % (success, result[-1]), "red")


def analyse(soundfile):
    (rate,signal) = wav.read(soundfile)
    mfcc_feat = mfcc(signal, rate)
    
    mean = np.mean(mfcc_feat, axis=0)
    var  = np.var(mfcc_feat, axis=0)
    X = np.append(mean, var)
    
    ''' write the features to the data structure to be sent to Azure '''
    counter = 0
    for key in ["mean%d" % x for x in range(NMFCCS)] + ["var%d"  % x for x in range(NMFCCS)] :
        data["Instance"]["FeatureVector"][key] = X[counter]
        counter += 1 


if __name__ == '__main__':
    ''' run a sample query with a Roesel's bush-cricket sound '''

    apply_workaround_bug11220()
    
    data["Instance"]["FeatureVector"]["label"] = "roesel"
    soundfile = '../Resources/cicadasounds/roesel_2013-10-21T21-45-30.399198.wav'
    
    analyse(soundfile)
    query()
    