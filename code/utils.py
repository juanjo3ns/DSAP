import numpy as np 
import cv2 as cv2


def load_image(path):
    #print(path)    
    a = np.array(cv2.imread(path + ".png"))
    #print(a.shape)
    #a = a.reshape(2,0,1)
    return a
