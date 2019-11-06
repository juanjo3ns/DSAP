import numpy as np
import cv2 as cv2


def load_image(path):
    return np.array(cv2.imread(path + ".png",0))
