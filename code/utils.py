import numpy as np
import cv2 as cv2

import mongodb_api as m


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