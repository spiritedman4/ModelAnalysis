import os.path

from flask import Flask,Blueprint,render_template
from project.flow_2 import models
from project.edas import eda
from project.forms import forms
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.types import String, Integer, Float, Boolean
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import backref
from flask_caching import Cache
from datetime import datetime
import multiprocessing
from multiprocessing import Process
from pathlib import Path
import threading
# from project.upload import uploading

app=Flask(__name__,static_folder = './project/static')
app.config['SECRET_KEY']="youcantseeme"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
current_path=Path(__file__).parent.absolute()
db_path=os.path.abspath(current_path/"project"/"databases")
app.config["SQLALCHEMY_DATABASE_URI"]= 'sqlite:///'+os.path.join(db_path,'ModelAnalysis.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['CACHE_TYPE']= "SimpleCache"
cache=Cache(app)
db= SQLAlchemy(app)



from flask import Flask
from celery import Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

db.init_app(app)

app.register_blueprint(forms,url_prefix="")
app.register_blueprint(models,url_prefix="")
app.register_blueprint(eda,url_prefix="")
# app.register_blueprint(uploading,url_prefix="")


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

"""-----------------------------------------------------------------------------------"""

if __name__ == "__main__":
    app.run(debug=True)
