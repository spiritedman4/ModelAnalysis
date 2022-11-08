import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from PIL import Image, ImageGrab
from io import BytesIO
import pandas as pd

from dev import celery


# Plot ROC curves
def plot_ROC(X, y, classifier, cv):
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp
    cv = StratifiedKFold(n_splits=cv)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    # figure = plt.figure()
    plt.gcf().clear()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png


#
# def plot_predVSreal(X, y, classifier, cv):
#     from sklearn.model_selection import cross_val_predict
#     # cross_val_predict returns an array of the same size as `y` where each entry
#     # is a prediction obtained by cross validation:
#     predicted = cross_val_predict(classifier, X, y, cv=cv)
#     plt.gcf().clear()
#     plt.scatter(y, predicted, edgecolors=(0, 0, 0))
#     plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
#     plt.xlabel('Measured')
#     plt.ylabel('Predicted')
#     from io import BytesIO
#     figfile = BytesIO()
#     plt.savefig(figfile, format='png')
#     figfile.seek(0)  # rewind to beginning of file
#     import base64
#     figdata_png = base64.b64encode(figfile.getvalue())
#     return figdata_png
#
#
# def plot_histsmooth(ds, columns):
#     sns.set()
#     plt.gcf().clear()
#     for col in columns:
#         sns.distplot(ds[col], label=col)
#     from io import BytesIO
#     plt.xlabel('')
#     plt.legend()
#     figfile = BytesIO()
#     plt.savefig(figfile, format='png')
#     figfile.seek(0)  # rewind to beginning of file
#     import base64
#     figdata_png = base64.b64encode(figfile.getvalue())
#     return figdata_png
#
#
# def plot_bar(ds, columns):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     sns.set()
#     plt.gcf().clear()
#     for col in columns:
#         sns.countplot(ds[col], label=col)
#     from io import BytesIO
#     plt.xlabel('')
#     plt.legend()
#     figfile = BytesIO()
#     plt.savefig(figfile, format='png')
#     # Show the plot
#     plt.show()
#     figfile.seek(0)  # rewind to beginning of file
#     import base64
#     figdata_png = base64.b64encode(figfile.getvalue())
#     return figdata_png
#
# # def plot_bar(ds, columns):
# #     import seaborn as sns
# #     import matplotlib.pyplot as plt
# #     sns.set()
# #     plt.gcf().clear()
# #     for col in columns:
# #         sns.countplot(ds[col],label=col)
# #         from io import BytesIO
# #     plt.xlabel('')
# #     plt.legend()
# #     figfile = BytesIO()
# #     plt.savefig(figfile, format='png')
# #     figfile.seek(0)  # rewind to beginning of file
# #     import base64
# #     figdata_png = base64.b64encode(figfile.getvalue())
# #     return figdata_png
#
#
# def plot_correlations(ds, corr, corrcat):
#     sns.set()
#     plt.gcf().clear()
#     if corrcat != '':
#         sns.pairplot(ds[corr], hue=corrcat)
#     else:
#         sns.pairplot(ds[corr])
#     from io import BytesIO
#     figfile = BytesIO()
#     plt.savefig(figfile, format='png')
#     figfile.seek(0)  # rewind to beginning of file
#     import base64
#     figdata_png = base64.b64encode(figfile.getvalue())
#     return figdata_png
#
#
# def plot_boxplot(ds, cat, num):
#     sns.set()
#     plt.gcf().clear()
#     with sns.axes_style(style='ticks'):
#         sns.factorplot(cat, num, data=ds, kind="box")
#     from io import BytesIO
#     plt.xlabel(cat)
#     plt.ylabel(num)
#     figfile = BytesIO()
#     plt.savefig(figfile, format='png')
#     figfile.seek(0)  # rewind to beginning of file
#     import base64
#     figdata_png = base64.b64encode(figfile.getvalue())
#     return figdata_png
#


@celery.task
def univariate_box_plot(df):
    fig_list = []
    from io import BytesIO
    from PIL import Image,ImageGrab
    import sys
    import base64
    global figure,stream,background
    figfile = BytesIO()
    columns=0
    print(df.dtypes)
    for i in df.columns:
        if df[i].dtype != 'O' and fig_list is None:
            figfile = BytesIO()
            col = i
            fig = plt.figure(figsize=(4,4))
            sns.boxplot(df[i], color='g')
            plt.style.use('ggplot')
            plt.xlabel(i.title(),fontsize=10)
            plt.savefig(figfile, format='png')
            figfile.seek(0)  # rewind to beginning of file
            import base64
            columns+=1
            fig_list = [figfile]
        elif df[i].dtype != 'O' and fig_list is not None:
            figfile = BytesIO()
            col = i
            fig = plt.figure(figsize=(4,4))
            sns.boxplot(df[i], color='g')
            plt.style.use('ggplot')
            plt.xlabel(i.title(),fontsize=10)
            plt.savefig(figfile, format='png')
            figfile.seek(0)  # rewind to beginning of file
            import base64
            columns+=1
            fig_list.append(figfile)



    images=[]
    for figfile in fig_list:
        if images is None:
            images=[Image.open(figfile)]
        else:
            images.append(Image.open(figfile))


    #method 1`-------------------------------------------------------------
    # widths, heights = zip(*(i.size for i in images))
    # total_width = sum(widths)
    # max_height = max(heights)
    # new_im = Image.new('RGB', (total_width, max_height))
    # x_offset = 0
    # for im in images:
    #     new_im.paste(im, (x_offset, 0))
    #     x_offset += im.size[0]
    #method 2--------------------------------------------------------------

    # min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    # imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in images))
    # output=BytesIO()
    # # imgs_comb = Image.fromarray(imgs_comb)
    # # imgs_comb.save(output,"PNG")
    # output.seek(0)
    # figdata_png_1=base64.b64encode(output.getvalue())
    # output.close()
    # return figdata_png_1

    # method 3------------------------------------------

    assert len(images) == columns
    a=int(columns/2)
    b=int(columns/2)
    w, h = images[0].size
    grid = Image.new('RGB', size=(a * w, b * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % a * w, i // b * h))

    output = BytesIO()
    output.seek(0)
    grid.save(output, "PNG")
    figdata_png_1 = base64.b64encode(output.getvalue())
    output.close()
    return figdata_png_1

    # img_w, img_h = zip(*(i.size for i in images))
    # background = Image.new('RGBA', (1300, 1300), (255, 255, 255, 255))
    # bg_w, bg_h = background.size
    # offset=0
    # for image in images:
    #     offset = (10, (((bg_h - img_h)) / 2) - 370)
    #     background.paste(image, (offset,0))
    #     offset += image.size[0]
    # output=BytesIO
    # output.seek(0)
    #
    # background.save(output,"PNG")
    # fig_data_png=base64.b64encode(output.getvalue())
    # output.close()
    # return fig_data_png

@celery.task
def univariate_dist_plot(df):
    sns.set()
    plt.gcf().clear()
    fig_list = []
    from io import BytesIO
    from PIL import Image, ImageGrab
    figfile = BytesIO()
    import sys
    import base64
    columns=0
    for i in df.columns:
        if df[i].dtype != 'O' and fig_list is None:
            figfile=BytesIO()
            col = i
            fig = plt.figure(figsize=(4,4))
            sns.distplot(df[i], color='b')
            plt.style.use('ggplot')
            plt.xlabel(i.title(),fontsize=10)
            plt.savefig(figfile, format='png')
            figfile.seek(0)  # rewind to beginning of file
            import base64
            columns+=1
            fig_list = [figfile]

        elif df[i].dtype != 'O' and fig_list is not None:
            figfile=BytesIO()
            col = i
            fig = plt.figure(figsize=(4,4))
            sns.distplot(df[i], color='b')
            plt.style.use('ggplot')
            plt.xlabel(i.title(),fontsize=10)
            plt.savefig(figfile, format='png')
            figfile.seek(0)  # rewind to beginning of file
            import base64
            columns+=1
            fig_list.append(figfile)

    images = []
    for figfile in fig_list:
        if images is None:
            images = [Image.open(figfile)]
        else:
            images.append(Image.open(figfile))

    assert len(images) == columns
    a = int(columns/2)
    b = int(columns/2)
    print(columns)

    w, h = images[0].size
    grid = Image.new('RGB', size=(a * w, b * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % a * w, i // b * h))

    output = BytesIO()
    output.seek(0)
    grid.save(output, "PNG")
    figdata_png_1 = base64.b64encode(output.getvalue())
    output.close()
    return figdata_png_1


@celery.task
def univariate_Histogram_plot(df):
    sns.set()
    plt.gcf().clear()
    fig_list = []
    from io import BytesIO
    from PIL import Image, ImageGrab
    figfile = BytesIO()
    import sys
    import base64
    columns=0
    for i in df.columns:
        if df[i].dtype != 'O' and fig_list is None:
            figfile=BytesIO()
            col = i
            fig = plt.figure(figsize=(4,4))
            plt.hist(df[i], color='r')
            plt.style.use('ggplot')
            plt.xlabel(i.title(),fontsize=10)
            plt.savefig(figfile, format='png')
            figfile.seek(0)  # rewind to beginning of file
            print(i)
            columns+=1
            fig_list = [figfile]

        elif df[i].dtype != 'O' and fig_list is not None:
            figfile = BytesIO()
            col = i
            fig=plt.figure(figsize=(4,4))
            plt.hist(df[i],color='r')
            plt.style.use('ggplot')
            plt.xlabel(i.title(),fontsize=10)
            plt.savefig(figfile,format='png')
            figfile.seek(0) # rewind to the beginning of the file
            print(i)
            columns += 1
            fig_list.append(figfile)


    images = []
    for figfile in fig_list:
        if images is None:
            images = [Image.open(figfile)]
        else:
            images.append(Image.open(figfile))
    print(columns)
    assert len(images) == columns
    a=int(2)
    b=int(columns/2)
    w, h = images[0].size
    print(w,h)
    grid = Image.new('RGB', size=(a * w , b * h))
    print(a * w, b * h)
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % a * w, i // b * h))
        print(i)
        print(i % a * w, i // b * h)


    output = BytesIO()
    output.seek(0)
    grid.save(output, "PNG")
    figdata_png_1 = base64.b64encode(output.getvalue())
    output.close()
    return figdata_png_1


def bivariate_analysis(df,target):
    sns.set()
    plt.gcf().clear()
    fig_list = []
    from io import BytesIO
    from PIL import Image, ImageGrab
    figfile = BytesIO()
    import sys
    import base64
    columns = 0
    print(target)
    print(df.dtypes)
    for i in df.columns:
        figfile = BytesIO()
        if df[i].dtype != 'O' and i != target and fig_list is None:
            print(i)
            col = i
            ov = pd.crosstab(df[col], df[target])
            plt.style.use('ggplot')
            ov.plot(kind='bar', figsize=(6,6), stacked=True)
            plt.xlabel(i.title())
            plt.savefig(figfile,format='png')
            figfile.seek(0)
            columns+=1
            fig_list=[figfile]

        elif df[i].dtype != 'O' and i != target and fig_list is not None:
            figfile = BytesIO()
            print(i)
            col = i
            ov = pd.crosstab(df[col], df[target])
            plt.style.use('ggplot')
            ov.plot(kind='bar', figsize=(6, 6), stacked=True)
            plt.xlabel(i.title())
            plt.savefig(figfile, format='png')
            figfile.seek(0)
            columns += 1
            fig_list.append(figfile)

    images = []
    for figfile in fig_list:
        if images is None:
            images = [Image.open(figfile)]
        else:
            images.append(Image.open(figfile))
    print(columns)
    assert len(images) == columns
    a = int(columns / 2)
    b = int(columns / 2)
    print(a,b)
    w, h = images[0].size
    grid = Image.new('RGB', size=(a * w, b * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % a * w, i // b * h))

    output = BytesIO()
    output.seek(0)
    grid.save(output, "PNG")
    figdata_png_1 = base64.b64encode(output.getvalue())
    output.close()
    return figdata_png_1