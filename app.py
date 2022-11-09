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


app=Flask(__name__,static_folder = './project/static')
app.config['SECRET_KEY']="youcantseeme"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
current_path=Path(__file__).parent.absolute()
db_path=os.path.abspath(current_path/"project"/"databases")
app.config["SQLALCHEMY_DATABASE_URI"]= 'sqlite:///'+os.path.join(db_path,'ModelAnalysis.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db= SQLAlchemy(app)



from flask import Flask

db.init_app(app)

app.register_blueprint(forms,url_prefix="")
app.register_blueprint(models,url_prefix="")

INPUT_PATH = os.path.join(os.path.dirname(__file__), "static")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "staticfiles")
SKIP_COMPRESS_EXTENSIONS = [
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    # Compressed files
    ".zip",
    ".gz",
    ".tgz",
    ".bz2",
    ".tbz",
    ".xz",
    ".br",
    # Flash
    ".swf",
    ".flv",
    # Fonts
    ".woff",
    ".woff2",
]


def remove_files(path):
    print(f"Removing files from {path}")
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def main():
    # remove all files from "staticfiles"
    remove_files(OUTPUT_PATH)

    for dirpath, dirs, files in os.walk(INPUT_PATH):
        for filename in files:
            input_file = os.path.join(dirpath, filename)
            with open(input_file, "rb") as f:
                data = f.read()
            # compress if file extension is not part of SKIP_COMPRESS_EXTENSIONS
            name, ext = os.path.splitext(filename)
            if ext not in SKIP_COMPRESS_EXTENSIONS:
                # save compressed file to the "staticfiles" directory
                compressed_output_file = os.path.join(OUTPUT_PATH, f"{filename}.gz")
                print(f"\nCompressing {filename}")
                print(f"Saving {filename}.gz")
                output = gzip.open(compressed_output_file, "wb")
                try:
                    output.write(data)
                finally:
                    output.close()
            else:
                print(f"\nSkipping compression of {filename}")
            # save original file to the "staticfiles" directory
            output_file = os.path.join(OUTPUT_PATH, filename)
            print(f"Saving {filename}")
            with open(output_file, "wb") as f:
                f.write(data)
main()

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
