import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from PIL import Image, ImageGrab
from io import BytesIO
import pandas as pd



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
