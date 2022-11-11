import logging
# suppress all warnings
import itertools
import logging
import os
import threading
import warnings

import joblib
# from pandas import Series, DataFrame
import numpy as np
import pandas as pd
from flask import make_response, Blueprint, session, flash,app
from datetime import timedelta
from flask import redirect, url_for
from flask import render_template, request

warnings.filterwarnings("ignore")
from pathlib import Path
import pdfkit
from queue import Queue

# blueprint
models = Blueprint("models", __name__, template_folder='/templates/application', static_folder='static')
# WHITENOISE_MAX_AGE = 31536000 if not app.config["DEBUG"] else 0

#     # configure WhiteNoise
#     app.wsgi_app = WhiteNoise(
#         app.wsgi_app,
#         root=os.path.join(os.path.dirname(__file__), "static"),
#         prefix="assets/",
#         max_age=WHITENOISE_MAX_AGE,
#     )
logging.basicConfig(filename="applicationlognew.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger('pydrop')
from IPython.display import HTML

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict

from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split  # import train_test_split function
from sklearn.linear_model import LogisticRegression  # import LogisticRegression
from sklearn.metrics import roc_auc_score  # import accuracy metrics
from sklearn.ensemble import RandomForestClassifier  # import RandomForestClassifier
import pickle
from deco import concurrent,synchronized

from functools import wraps
# class Modeluidingvariables():
#     def __init__(self, file, df,df_preview_table, fieldnames, df1, algorithms, graphJSON, result, msg):
#         self.file = file
#         self.df = df
#         self.df_preview_table=df_preview_table
#         self.fieldnames = fieldnames
#         self.df1 = df1
#         self.algorithms = algorithms
#         self.result = result
#         self.msg = msg

import sys
folder_path = Path(__file__).parent.absolute()
folder = os.path.abspath(folder_path / "uploaded_csv_files")
print(folder_path,folder)
print(folder)
sys.stdout.flush()
uploaded_csv_files = os.listdir(folder)

from threading import Thread


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return





# GET THE BASIC DESCRIPTIVE STATISTICSs    # summary route 3
def get_descriptive_statistics(data):
    data = data.select_dtypes(include=np.number)
    mean = pd.DataFrame(data.apply(np.mean)).T
    median = pd.DataFrame(data.apply(np.median)).T
    std = pd.DataFrame(data.apply(np.std)).T
    min_value = pd.DataFrame(data.apply(min)).T
    max_value = pd.DataFrame(data.apply(max)).T
    range_value = pd.DataFrame(
        data.apply(lambda x: x.max() - x.min())).T
    skewness = pd.DataFrame(data.apply(lambda x: x.skew())).T
    kurtosis = pd.DataFrame(data.apply(lambda x: x.kurtosis())).T
    global summary_stats, result_summary_stats, summary_stats_attributes
    summary_stats = pd.concat(
        [min_value, max_value, range_value, mean, median, std, skewness, kurtosis]).T.reset_index()
    summary_stats.columns = ['Attributes', 'Min', 'Max', 'Range', 'Mean', 'Median', 'Std', 'Skewness',
                             'Kurtosis']
    summary_stats_attributes = summary_stats['Attributes'].tolist()
    result_summary_stats = HTML(summary_stats.to_html())
    return result_summary_stats


def get_categorical_descriptive_statistics(data):
    global categorical_summary_stats
    data = data.select_dtypes(include=['object'])
    if not data.empty:
        summary_stats_1 = data.describe()
        categorical_summary_stats_table = HTML(summary_stats_1.to_html())
        categorical_summary_stats = {
            'data': categorical_summary_stats_table
        }
    else:
        categorical_summary_stats = {}
    return categorical_summary_stats


def find_missing_values(data):
    global missing_data_information,message,missing_data_information_table,missing_columns
    total = data.isnull().sum().sort_values(ascending=False)  # compute the total number of missing values
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(
        ascending=False)  # compute the percentage of missing values
    missing_data = pd.concat([total, percent], axis=1,
                             keys=['Total', 'Percent'])  # add all information to one dataframe
    missing_data = missing_data[
        missing_data['Total'] > 0]  # filter the dataframe to only the features with missing values
    message = "No missing data found. Please proceed to datapreprocessing"
    missing_columns = list(missing_data.index)
    missing_data_information_table = HTML(missing_data.to_html())
    if missing_data.empty:
        missing_data_information = {
            "message": message
        }
    else:
        missing_data_information = {
            "missing_data_information_table": missing_data_information_table
        }

    return missing_data_information, missing_columns


def removeNulls(dataframe, axis, percent):  # 2
    global msg,df_null_cleared
    '''
    * removeNull function will remove the rows and columns based on parameters provided.
    * dataframe : Name of the dataframe
    * axis      : axis = 0 defines drop rows, axis =1(default) defines drop columns
    * percent   : percent of data where column/rows values are null,default is 0.3(30%)

    '''
    df_null_cleared = dataframe.copy()
    ishape = df_null_cleared.shape
    if axis == 0:
        rownames = df_null_cleared.transpose().isnull().sum()
        rownames = list(rownames[rownames.values > percent * len(df_null_cleared)].index)
        df_null_cleared.drop(df.index[rownames], inplace=True)
        print("\nNumber of Rows dropped\t: ", len(rownames))
    else:
        colnames = (df_null_cleared.isnull().sum() / len(df_null_cleared))
        print(type(colnames))
        print(type(percent))
        colnames = list(colnames[colnames.values >= float(percent)].index)
        df_null_cleared.drop(labels=colnames, axis=1, inplace=True)
        print("Number of Columns dropped\t: ", len(colnames))

    msg = ["Removed null values successfully", f"Old dataset rows,columns {ishape}",
           f"New dataset rows,columns {df_null_cleared.shape}"]

    print("\nOld dataset rows,columns", ishape, "\nNew dataset rows,columns", df_null_cleared.shape)
    session["df_null_cleared"]="df_null_cleared"
    return df_null_cleared


# To Impute the missing values based on mean,median and mode

def conditional_impute(df, column_name, choice):
    global df_preprocessed, alert
    try:
        if choice == 'mean':
            mean_value = df[column_name].mean()
            df[column_name].fillna(value=mean_value, inplace=True)
        elif choice == 'median':
            median_value = df[column_name].median()
            df[column_name].fillna(value=median_value, inplace=True)
        elif choice == 'mode':
            mode_value = df[column_name].mode()[0]
            df[column_name].fillna(value=mode_value, inplace=True)
    except Exception:
        print('Wrong Argument')
    else:
        alert = f"{column_name.upper()} is imputed with {choice.upper()} method."
    df_preprocessed = df
    find_missing_values(df_preprocessed)
    return df_preprocessed, alert

def create_dataframe(csv_file):
    global df, df_preview_table,df_null_cleared,df_preprocessed
    ''' folder path for reading the files!'''
    folder_path = Path(__file__).parent.absolute()
    folder = os.path.abspath(folder_path / "uploaded_csv_files")
    uploaded_csv_files = os.listdir(folder)
    try:
        del df_null_cleared
        print("this")
    except NameError or UnboundLocalError:
        pass

    try:
        del df_preprocessed
        print("this-2")

    except NameError or UnboundLocalError:
        pass

    df = pd.read_csv(f"{folder}\\{csv_file}", header='infer',error_bad_lines=False)

    # session['df_dictionary']=df_dictionary
    logging.info(f"Dataframe for {csv_file} is created!")

    df_preview = df.head(5)
    df_preview_table = HTML(df_preview.to_html())
    return df, df_preview_table, uploaded_csv_files


# def remove_outliers(data):
#     num_train = data.select_dtypes(include=["number"])
#     cat_train = data.select_dtypes(exclude=["number"])
#     from scipy import stats
#     idx = np.all(stats.zscore(num_train) < 3, axis=1)
#     train_cleaned = pd.concat([num_train.loc[idx], cat_train.loc[idx]], axis=1)
#     print(train_cleaned)
#     return train_cleaned


def get_metrics_data(Tasks, Metrics, db):
    global metrics, metrics_data_list, metrics_data
    metrics = db.session.query(Tasks, Metrics).outerjoin(Metrics, Tasks.task_name == Metrics.task_name)
    metrics_data_list = []
    for metric in metrics:
        if metric[1]:
            metrics_data = {}
            metrics_data['task_id'] = metric[0].task_id
            metrics_data['task_name'] = metric[0].task_name
            metrics_data['time_created'] = metric[0].time_created
            metrics_data['task_status'] = metric[0].task_status
            metrics_data['model_name'] = metric[1].model_name
            metrics_data['accuracy'] = metric[1].accuracy
            metrics_data['accuracy_cv_score'] = metric[1].accuracy_cv_score
            metrics_data['accuracy_cv_stddev'] = metric[1].accuracy_cv_stddev
            metrics_data['Train_Accuracy'] = metric[1].Train_Accuracy
            metrics_data['Test_Accuracy'] = metric[1].Test_Accuracy
            metrics_data['precision_score'] = metric[1].precision_score
            metrics_data['recall_score'] = metric[1].recall_score
            metrics_data['f1_score'] = metric[1].f1_score
            metrics_data['roc_auc_score'] = metric[1].roc_auc_score
            metrics_data_list.append(metrics_data)
    return metrics_data_list


def get_tasks(all_tasks):
    global tasks_list
    tasks_list = []
    for task in all_tasks:
        task_data = {}
        task_data['task_id'] = task.task_id
        task_data['task_name'] = task.task_name
        task_data['time_created'] = task.time_created
        task_data['time_closed'] = task.time_closed
        task_data['task_status'] = task.task_status
        tasks_list.append(task_data)
    return tasks_list


# task route
@models.route('/task', methods=["GET", "POST"])
def create_task():
    STATE = ('INACTIVE', 'ACTIVE')
    from dev import db
    from dev import Tasks_new, Metrics
    all_tasks = Tasks_new.query.all()
    get_metrics_data(Tasks_new, Metrics, db)
    get_tasks(all_tasks)

    if request.method == "POST":
        task_name = request.form.get("task_name")
        task_names = Tasks_new.query.filter_by(task_name=task_name).first()
        if task_names is None:
            task = Tasks_new(task_name=task_name)
            flash('Task is created', 'success')
            session[task_name] = task_name
            db.session.add(task)
            db.session.commit()
            get_tasks(all_tasks)
        else:
            flash('Tasks already exists!. Please provide an another name.')

        return render_template('application/task_flow.html', task_name=task_name, metrics_data_list=metrics_data_list,
                               STATE=STATE, tasks_list=tasks_list)
    return render_template('application/task_flow.html', metrics_data_list=metrics_data_list, STATE=STATE,
                           tasks_list=tasks_list)


@models.errorhandler(500)
def internal_error(e):
    return render_template('errors/500.html'), 500


@models.route('/<task_name>/upload', methods=["GET", "POST"])
def index(task_name):
    global file
    uploaded_csv_files = os.listdir(folder)
    task_name = task_name
    if request.method == "POST":
        file = request.files['file']
        save_path = os.path.join(folder, secure_filename(file.filename))

        current_chunk = int(request.form['dzchunkindex'])

        # If the file already exists it's ok if we are appending to it,
        # but not if it's new file that would overwrite the existing one
        if os.path.exists(save_path) and current_chunk == 0:
            # 400 and 500s will tell dropzone that an error occurred and show an error
            return make_response(('File already exists! Please click submit to proceed further.', 400))

        try:
            with open(save_path, 'ab') as f:
                print("3")
                f.seek(int(request.form['dzchunkbyteoffset']))
                f.write(file.stream.read())

        except OSError:
            print("4")
            # log.exception will include the traceback so we can see what's wrong
            log.exception('Could not write to file')
            return make_response(("Not sure why,"
                                  " but we couldn't write the file to disk", 500))

        total_chunks = int(request.form['dztotalchunkcount'])

        if current_chunk + 1 == total_chunks:
            print("5")
            # This was the last chunk, the file should be complete and the size we expect
            if os.path.getsize(save_path) != int(request.form['dztotalfilesize']):
                print("6")
                log.error(f"File {file.filename} was completed, "
                          f"but has a size mismatch."
                          f"Was {os.path.getsize(save_path)} but we"
                          f" expected {request.form['dztotalfilesize']} ")
                return make_response(('Size mismatch', 500))

            else:
                log.info(f'File {file.filename} has been uploaded successfully')
        else:
            print("8")
            log.debug(f'Chunk {current_chunk + 1} of {total_chunks} '
                      f'for file {file.filename} complete')

    print("upload done")
    return render_template('application/new_upload_flow.html', uploaded_csv_files=uploaded_csv_files,
                           task_name=task_name)


@models.route('/<task_name>/datapreview', methods=['GET', 'POST'])
def data_preview(task_name):
    global file, df_null_cleared, df_preview_table, filename
    if request.method == "POST":
        return redirect(url_for('models.index', task_name=task_name))
    filename = file.filename
    '''creating the dataframe'''
    create_dataframe(filename)
    if df_preview_table is None:
        df_preview_table = session['df_preview_table']
    else:
        df_preview_table = df_preview_table

    return render_template("application/new_upload_flow.html", df_preview=df_preview_table, filename=filename,
                           uploaded_csv_files=uploaded_csv_files, task_name=task_name)


@models.route('<task_name>/datapreview_1', methods=["GET", "POST"])
def data_preview_1(task_name):
    global uploaded_file_selected, uploaded_csv_files, df_preview_table, filename
    task_name = task_name
    filename = request.form.get('uploaded_csv_files')
    create_dataframe(filename)
    if df_preview_table is None:
        df_preview_table = session['df_preview_table']
    else:
        df_preview_table = df_preview_table

    ''' folder path for reading the files!'''
    folder_path = Path(__file__).parent.absolute()
    folder = os.path.abspath(folder_path / "uploaded_csv_files")
    uploaded_csv_files = os.listdir(folder)
    return render_template("application/new_upload_flow.html", df_preview=df_preview_table, filename=filename,
                           uploaded_csv_files=uploaded_csv_files, task_name=task_name)


@models.route("/<task_name>/datasummary", methods=['POST', 'GET'])
def data_summary(task_name):
    global missing_data, msg, df_null_cleared, result, filename, df
    if request.method == "POST":
        threshold = request.form.get("threshold")
        axis = request.form.get("axis")
        removeNulls(dataframe=df, axis=axis, percent=threshold)
        find_missing_values(data=df_null_cleared)
        return render_template('application/data_summary_flow.html', msg=msg,
                               missing_data_information=missing_data_information, task_name=task_name,
                               filename=filename)
    try:
        find_missing_values(df)
    except NameError:
        return render_template('application/data_summary_flow.html',task_name=task_name)

    return render_template('application/data_summary_flow.html', missing_data_information=missing_data_information,
                           task_name=task_name, filename=filename)


@models.route("/<task_name>/datapreprocessing", methods=["GET", "POST"])
def datapreprocessing(task_name):
    global column_names
    alerts = []
    results = []
    col_list_1 = []
    dict_new = {}


    if request.method == "POST":
        for i, e in enumerate(column_names):
            j = request.form.get(e)
            if j is not None:
                dict_new[e] = j



        try:
            try:
                for key, value in dict_new.items():
                    from multiprocessing.pool import ThreadPool
                    pool = ThreadPool(processes=5)

                    async_result = pool.apply_async(conditional_impute,
                                                    (df_preprocessed, key, value))  # tuple of args for foo

                    # do some other stuff in the main process

                    return_val = async_result.get()
                    results.append(return_val)
            except NameError:
                for key, value in dict_new.items():
                    from multiprocessing.pool import ThreadPool
                    pool = ThreadPool(processes=5)

                    async_result = pool.apply_async(conditional_impute,(df_null_cleared, key, value))  # tuple of args for foo

                    # do some other stuff in the main process

                    return_val = async_result.get()
                    results.append(return_val)

            # conditional_impute(df_null_cleared, key, value)
            # alerts.append(alert)
        except NameError:
            for key, value in dict_new.items():
                from multiprocessing.pool import ThreadPool
                pool = ThreadPool(processes=5)

                async_result = pool.apply_async(conditional_impute,(df, key, value))  # tuple of args for foo

                # do some other stuff in the main process

                return_val = async_result.get()
                results.append(return_val)
            # conditional_impute(df, key, value)
            # alerts.append(alert)
    try:
        get_descriptive_statistics(data=df_null_cleared)
        get_categorical_descriptive_statistics(data=df_null_cleared)
        column_names = df_null_cleared.columns
    except NameError:
        try:
            get_descriptive_statistics(data=df)
            get_categorical_descriptive_statistics(data=df)
            column_names = list(df.columns)
        except NameError:
            return render_template('application/data_preprocessing_flow.html',task_name=task_name)
    results=[x[1] for x in results]
    return render_template('application/data_preprocessing_flow.html', result_summary_stats=result_summary_stats,
                           task_name=task_name, missing_columns=missing_columns,
                           categorical_summary_stats=categorical_summary_stats, alerts=results)


@models.route("/<task_name>/eda", methods=['POST', 'GET'])
def eda(task_name):
    global figs, target, fieldnames, ds , check_var
    figs = {}
    if request.method == "GET":
        try:
            try:
                fieldnames = df_preprocessed.columns
                ds=df_preprocessed
                check_var = "df_check-pre"
                print(1)
            except NameError:
                fieldnames=df_null_cleared.columns
                check_var="df_check-null"
                ds=df_null_cleared
                print(2)
        except NameError:
            try:
                fieldnames = df.columns
                ds = df
                check_var='normal-df'
                print(3)
            except NameError:
                print(4)
                return render_template("application/new_eda_flow.html", task_name=task_name)

        return render_template("application/new_eda_flow.html", fieldnames=fieldnames, task_name=task_name)
    if request.method == "POST":
        ind = request.form.get("indepvarmultiple")
        dep = request.form.get("depvarmultiple")
        indep_var = request.form.getlist('indepvar')
        dep_var = request.form.getlist('depvar')
        if dep_var:
            target = dep_var[0]
        print(check_var)
        import project.plotfunction as plotfunc
        figs['Box Plot'] = str(plotfunc.univariate_box_plot(ds), 'utf-8')
        figs['Dist Plot'] = str(plotfunc.univariate_dist_plot(ds), 'utf-8')
        figs['Histogram']= str(plotfunc.univariate_Histogram_plot(ds), 'utf-8')
        # figs['Bivariate Analysis']= str(plotfunc.bivariate_analysis(ds, target), 'utf-8')


    return redirect(url_for('models.show_graphs', task_name=task_name))

    # return render_pdf(url_for('drawgraphs.html', figs=figs))


@models.route("/<task_name>/graphs", methods=["GET", "POST"])
def show_graphs(task_name):
    global rendered
    rendered = render_template('application/graphs_new.html', figs=figs, task_name=task_name)
    return render_template('application/graphs_flow_new.html', figs=figs, task_name=task_name)


@models.route("/<task_name>/graphs-pdf", methods=["GET", "POST"])
def create_pdf(task_name):
    pdfkit.from_url(rendered, "C:\\Users\\srinivasan.b\\PycharmProjects\\ModelAnalysis\\pdfs", f"{task_name}.pdf")
    return rendered


@models.route("/<task_name>/model_building", methods=["GET", "POST"])
def model_building(task_name):
    global df_all_models_preview_table, model_building_variables, indep_variables, dep_variables, pkl_files_dict, df_model_stats_ranked
    pkl_files_dict = {}
    if request.method == "POST":
        global df_all_models_preview_table
        indep_variables = request.form.getlist('indepvar')
        dep_variables = request.form.getlist('depvar')
        test_size = request.form.get('train_test_value')
        X = pd.DataFrame()
        for i in indep_variables:
            if i in model_building_variables:
                X[i] = ds[i]
            else:
                msg = "Please select the parameters"
        Y = pd.DataFrame()
        for j in dep_variables:
            if j in model_building_variables:
                Y[j] = ds[j]
            else:
                msg = "Please select the parameters"

        models = {'LogisticRegression': LogisticRegression(random_state=0),
                  'DecisionTree': DecisionTreeClassifier(random_state=0),
                  'RandomForest': RandomForestClassifier(random_state=0)}

        from dev import db
        from dev import Metrics

        def output_model_stats(mname, minst):
            global model_stats
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
            model_stats = {}
            model_scores = []

            # create pipeline with scaler and model instantiation steps
            model_pipe = make_pipeline(StandardScaler(), minst)

            # fit training data and run model score
            model_new = model_pipe.fit(X_train, y_train)
            y_pred_test = model_new.predict(X_test)
            from sklearn.metrics import accuracy_score
            Model_Accuracy_Score = (accuracy_score(y_test, y_pred_test))
            model_scores.append(round(Model_Accuracy_Score, 2))

            scores = cross_val_score(model_new, X_train, y_train, cv=10, scoring='accuracy')
            model_scores.append(round(scores.mean(), 2))
            model_scores.append(round(scores.std(), 2))

            train_preds = model_new.predict(X_train)
            test_preds = model_new.predict(X_test)

            # evaluate
            train_accuracy = roc_auc_score(y_train, train_preds)
            test_accuracy = roc_auc_score(y_test, test_preds)
            train_accuracy = round(train_accuracy, 2)
            test_accuracy = round(test_accuracy, 2)

            score_logreg = [minst, train_accuracy, test_accuracy]
            models = pd.DataFrame([score_logreg])
            folder_path = Path(__file__).parent.absolute()
            folder = os.path.abspath(folder_path / 'generated_pickle_files')

            path = (f"{folder}\\{task_name}")
            if not os.path.exists(path):
                os.makedirs(path)

            def unique_file(basename, ext):
                global c
                actualname = "%s.%s" % (basename, ext)
                c = itertools.count()
                while os.path.exists(actualname):
                    actualname = "%s (%d).%s" % (basename, next(c), ext)
                    pkl_files_dict[mname] = actualname
                return actualname

            with open(unique_file((f"{path}\\model_{mname}_{task_name}"), 'pkl'), 'wb') as f:
                pickle.dump(model_new, f)
            model_scores.append(train_accuracy)
            model_scores.append(test_accuracy)

            # implement cross validation predictions on training data
            y_train_cv_pred = cross_val_predict(model_new, X_train, y_train, cv=10)

            # calculate precision and recall
            p = precision_score(y_train, y_train_cv_pred)
            r = recall_score(y_train, y_train_cv_pred)
            model_scores.append(round(p, 2))
            model_scores.append(round(r, 2))

            # calculate F1 score
            f1_score = 2 * (p * r) / (p + r)
            model_scores.append(round(f1_score, 2))

            # calculate ROC AUC score using cross_val_score
            roc_auc_cvs = cross_val_score(model_new, X_train, y_train, cv=10, scoring='roc_auc').mean()
            model_scores.append(round(roc_auc_cvs, 2))

            # create dictionary key/pair value
            model_stats[mname] = model_scores
            new_metrics = Metrics(task_name=task_name, model_name=mname, accuracy=model_scores[0],
                                  accuracy_cv_score=model_scores[1], accuracy_cv_stddev=model_scores[2],
                                  Train_Accuracy=model_scores[3], Test_Accuracy=model_scores[4],
                                  precision_score=model_scores[5], recall_score=model_scores[6],
                                  f1_score=model_scores[7],
                                  roc_auc_score=model_scores[7])
            db.session.add(new_metrics)
            db.session.commit()

            colnames = ['accuracy', 'accuracy_cv_score', 'accuracy_cv_stddev', 'Train_Accuracy', 'Test_Accuracy',
                        'precision_score', 'recall_score', 'f1_score',
                        'roc_auc_score (cross_val_score)']

            # put model stats into a dataframe
            df_model_stats = pd.DataFrame.from_dict(model_stats, orient='index', columns=colnames)
            df_model_stats_ranked = df_model_stats.sort_values(by='accuracy', ascending=False)
            # store accuracy in a new dataframe
            print("1")
            # output is a dataframe
            return df_model_stats_ranked

        #      multiprocessing
        results = []
        for key, value in models.items():
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(processes=5)
            print(pool)

            async_result = pool.apply_async(output_model_stats, (key, value))  # tuple of args for foo

            # do some other stuff in the main process

            return_val = async_result.get()
            results.append(return_val)
        df_model_stats_ranked = pd.concat(results)

        df_all_models_preview_table = HTML(df_model_stats_ranked.to_html())

        return render_template('application/model_building_flow.html', task_name=task_name,
                               fieldnames=model_building_variables,df_all_models_preview_table=df_all_models_preview_table)

    try:
        model_building_variables = ds.columns
        return render_template('application/model_building_flow.html', task_name=task_name,
                               fieldnames=model_building_variables)
    except:
        return render_template('application/model_building_flow.html', task_name=task_name)


@models.route("/<task_name>/prediction", methods=["GET", "POST"])
def prediction(task_name):
    global indep_variables, pkl_file_path,prediction_result
    prediction_variables_selected = []
    # indep_variables=[x.upper() for x in indep_variables]
    if request.method == "POST":
        # from dev import Tasks_new,db
        # task=db.session.query(Tasks_new).filter_by(task_name=task_name).one()
        # if task:
        #     task.task_status=0
        #     task.time_closed=datetime.datetime.utcnow()
        #     db.session.commit()
        for i in indep_variables:
            prediction_variables = float(request.form.get(i))
            prediction_variables_selected.append(prediction_variables)
        algorithm = request.form.get('algorithm')
        for key, value in pkl_files_dict.items():
            if algorithm in pkl_files_dict:
                pkl_file_path = pkl_files_dict[algorithm]
        model_file = open(pkl_file_path, 'rb')
        model = joblib.load(model_file)
        prediction = [prediction_variables_selected]
        model_prediction = model.predict(prediction)
        prediction_values=['Not Default','Default']
        prediction_result= prediction_values[int(model_prediction)]
        return render_template('application/prediction_flow.html', task_name=task_name, indep_variables=indep_variables,prediction_result=prediction_result)


    try:
        return render_template('application/prediction_flow.html', task_name=task_name, indep_variables=indep_variables)

    except NameError:
        return render_template('application/prediction_flow.html', task_name=task_name)
