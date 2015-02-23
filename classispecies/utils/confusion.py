from collections import OrderedDict
from classispecies import settings

if not settings.is_mpl_backend_set():
    settings.set_mpl_backend()

from matplotlib import pyplot as plt
import numpy as np


class ConfusionMatrix(OrderedDict):
    def __init__(self, classes, class_names=None, *args, **kwds):
        OrderedDict.__init__(self, *args, **kwds)
        for class_ in classes:
            self[str(class_)] = OrderedDict()
            for class__ in classes:
                self[str(class_)][str(class__)] = 0
        self[-1] = {}
        self[-1][-1] = 0
        
        self.classes = classes
        self.nclasses = len(classes)
        self.class_names = class_names or classes
        
 
    def __repr__(self):
        del self[-1]
        out = ("    " + "%3s  " * self.nclasses +"\n") % tuple(self.classes)
        for nclass in range(len(self.classes)):
            y = self.values()[nclass].values()
            out += ("%3s " + "%.2f " * self.nclasses +"\n") % tuple([self.class_names[nclass]] +
                list([float(x) if sum(y) > 0 else 0 for x in y])) 
        self[-1] = {}
        self[-1][-1] = 0
        return out
    
    def toval(self):
        del self[-1]
        out = ("    " + "%3s  " * self.nclasses +"\n") % tuple(self.classes)
        for nclass in range(len(self.classes)):
            y = self.values()[nclass].values()
            out += ("%3s " + "%.2f " * self.nclasses +"\n") % tuple([self.class_names[nclass]] +
                list([float(x)/sum(y) if sum(y) > 0 else 0 for x in y])) 
        self[-1] = {}
        self[-1][-1] = 0
        return out
 
    def get_value(self, x, y):
        if sum(self[str(x)].values()) == 0:
            return 0
        else:
            return float(self[str(x)][str(y)])/sum(self[str(x)].values())
            #return float(self[str(x)][str(y)])
        
    def add(self, true, pred):
        ''' add a prediction `pred` for the class `true` '''
        self[str(true)][str(pred)] += 1
 
    def plot(self, outputname=None):
        rcdef = plt.rcParams.copy()
        matplotlib.rcParams.update({
                                    "savefig.bbox" : "tight",
                                    "pgf.texsystem": "pdflatex",
                                    "font.family": "Times New Roman",
                                    "font.serif": ["Times"],
                                    "font.size" : 27})
        fig = plt.figure(figsize=(self.nclasses,self.nclasses))
        for i in range(self.nclasses):
            for j in range(self.nclasses):
                val = self.get_value(self.classes[i],self.classes[j])
                str_val = str(int(val)) if int(val) == val else str("%.2f" % val)[1:]
 
                plt.broken_barh([(j, 1)], (i, 1), facecolors=(1-val, 1-val, 1-val), hold=True)
                plt.text(j+0.5,i+0.5, str_val, color='white' if val > 0.5 else 'black',
                         horizontalalignment='center',
                         verticalalignment='center',)
 
        ax = plt.gca()
        ax.set_xticks(np.arange(self.nclasses) + 0.5)
        ax.set_yticks(np.arange(self.nclasses) + 0.5)
        ax.set_xticklabels([x.upper() for x in self.class_names])
        ax.set_yticklabels([x.upper() for x in self.class_names])
        ax.invert_yaxis()
        plt.tick_params(\
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            labelbottom='off', # labels along the bottom edge are off
            labeltop='on',
            labelleft='on',)
 
        fig.savefig(outputname or "confusion.pdf")
        plt.clf()
        plt.rcParams.update(rcdef)
        
    def html(self):
        
        out = '''\
<table class="confusion">
    <tr>
        <th></th>
'''
        vals = self.class_names[:]
        out += "\n".join(["        <th>%s</th>" % s for s in self.class_names]) + '''
    </tr>
'''
    
        for i in range(self.nclasses):
            out += "    <tr>\n        <th>%s</th>\n" % self.class_names[i]
            for j in range(self.nclasses):
                
                val = self.get_value(self.classes[i],self.classes[j])
                str_val = str(int(val)) if int(val) == val else str("%.2f" % val)[1:]
                facecolor='white' if val > 0.5 else 'black'
                bgcolor=tuple(np.round(255*np.array([1-val, 1-val, 1-val])).astype(int))
                
                out += '        <td style="color: %s; background: rgb%s">%.2f</td>\n' % (facecolor,  bgcolor, val)
 
            out += "    </tr>\n"
        out += "</table>\n"
        return out
 