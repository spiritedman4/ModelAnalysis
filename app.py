import os.path

from flask import Flask,Blueprint,render_template
from flow_2 import models
from forms import forms
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.types import String, Integer, Float, Boolean
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import backref
from datetime import datetime
import multiprocessing
from multiprocessing import Process
from pathlib import Path
import threading
from whitenoise import WhiteNoise
import logging


app=Flask(__name__,static_folder = './project/static')
app.config['SECRET_KEY']="youcantseeme"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
current_path=Path(__file__).parent.absolute()
db_path=os.path.abspath(current_path/"project"/"databases")
# app.config["SQLALCHEMY_DATABASE_URI"]= 'sqlite:///'+os.path.join(db_path,'ModelAnalysis.db')
app.config["SQLALCHEMY_DATABASE_URI"]="postgresql://xgmddzrxkditty:86bacb5328b6ca3afdff388afdf75f963d363040ddb1213e1a66134395ffa239@ec2-3-217-251-77.compute-1.amazonaws.com:5432/dcibpuoa0ue5g3"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db= SQLAlchemy(app)



from flask import Flask

db.init_app(app)

app.register_blueprint(forms,url_prefix="")
app.register_blueprint(models,url_prefix="")

import os

app.wsgi_app = WhiteNoise(app.wsgi_app,
        root=os.path.join(os.path.dirname(__file__), 'static'),
        prefix='static/')

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
"""-----------------------------------------------------------------------------------"""

class Tasks_new(db.Model):
    __tablename__="Tasks_new"
    task_id=db.Column(db.Integer,primary_key=True)
    task_name=db.Column(db.String(30),unique=True)
    time_created = db.Column(db.DateTime, default=datetime.utcnow())
    time_closed=db.Column(db.DateTime,default=None)
    task_status=db.Column(db.Boolean,default=1)

class Metrics(db.Model):
    __tablename__="Metrics"
    id=db.Column(db.Integer,primary_key=True)
    task_name=db.Column(db.String,db.ForeignKey('Tasks_new.task_name'))
    Tasks_new = db.relationship("Tasks_new", backref=backref("Tasks_new", uselist=False))
    model_name=db.Column(db.String(40))
    accuracy=db.Column(db.Float)
    accuracy_cv_score=db.Column(db.Float)
    accuracy_cv_stddev=db.Column(db.Float)
    Train_Accuracy=db.Column(db.Float)
    Test_Accuracy=db.Column(db.Float)
    precision_score=db.Column(db.Float)
    recall_score=db.Column(db.Float)
    f1_score=db.Column(db.Float)
    roc_auc_score=db.Column(db.Float)
        
db.create_all()
"""-----------------------------------------------------------------------------------"""

if __name__ == "__main__":
    app.run(debug=True)
