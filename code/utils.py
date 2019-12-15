import numpy as np
import cv2 as cv2
from IPython import embed
import mongodb_api as m
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_image(path):
    return np.array(cv2.imread(path + ".png",0))


def frequency(filterr={"split":"train"}, coll="task5"):
    a = m.get_from(filt=filterr, collection=coll)
    suma_high = np.zeros(8)
    suma_low = np.zeros(23)

    for i, it in enumerate(a):
        suma_high += np.array(it["high_labels"])
        suma_low += np.array(it["low_labels"])

    frq_high = suma_high/np.sum(suma_high)
    frq_low = suma_low/np.sum(suma_low)
    f1 = np.array(np.ones(8) - frq_high)
    f2 = np.array(np.ones(23) - frq_low)
    f8 = f1/np.sum(f1)
    f23 = f2/np.sum(f2)
    return f8, f23


def get_confusion_matrix(confusion_matrix):

    fig, ax= plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, ax = ax, fmt='g')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(labels)
    # ax.yaxis.set_ticklabels(labels)
    return fig
