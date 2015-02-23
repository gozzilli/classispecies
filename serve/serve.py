# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:09:20 2015

@author: dz2v07
"""

from flask import Flask, render_template, Blueprint
from pymongo import MongoClient
import pymongo

app = Flask(__name__)
cs = Blueprint('cs', __name__,
                        template_folder='templates')
app.debug = True

client = MongoClient()
db = client.classispecies ## this is the database

    
@cs.route('/show/<model>/<guid>')
def serve_byid(model, guid):
    
    rows = db[model].find({"run_key" : guid}).sort("f1", pymongo.DESCENDING)
    if rows.count() == 0:
        return "<html><h4>Sorry, there is no model run with this key.</h4></html>"
    else:
        return render_template("results.html", rows=rows)

@cs.route('/show/<model>')
def serve_results(model):
    rows = db[model].find().sort("f1", pymongo.DESCENDING)
    return render_template("results.html", rows=rows)


@cs.route('/show/<model>/all')
def serve_all(model):
    
    rows = db[model].find()
    return render_template("results_all.html", rows=rows)


@cs.route('/ukorth/all')
def ukorth_all():
    return serve_all("ukorthoptera")

@cs.route('/ukorth')
def ukorth():    
    return serve_results("ukorthoptera")

@cs.route('/collected')
def collected():    
    return serve_results("collected")

@cs.route("/custom")
def custom():
    #rows = db.nfc_vs_rbc.find({"extract_mel" : False, "extract_dolog" : False, "extract_dct" : False, "sec_segments" : 0.5}, {"f1m": 0, "rocm" : 0})
    rows = db.nfc3species.find({"extract_mel" : False, 
                               "extract_dolog" : False, 
                               "extract_dct" : False, 
                               "sec_segments" : None,
                               "classifier" : "randomforest"})
#     rows = db.nfc_vs_rbc.find({"sec_segments" : None})
    return render_template("results.html", rows=rows)

@cs.route('/report/<model>/last')
def report_last(model):
    
    row = db[model].find_one()
    return render_template("simulationrun.html", **row)

@cs.route('/')
def home():
    
    return render_template("home.html", models=db.collection_names(False))

if __name__ == '__main__':
    app.debug = True
    app.register_blueprint(cs)
    app.run()
    
