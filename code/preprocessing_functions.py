import scipy, pylab
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import mongodb_api as mongo
#from __future__ import division
from numpy import fft 
from IPython import embed
#from pytfd import helpers as h 


def stft(x, fs=8000, framesz=1, hop=1, window="hamming", norm=True):#framesz and hope in time domain
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)   #Overlaping factor

    #Select window
    if window=="hamming":
        w = scipy.hamming(framesamp)
    else:
        w = scipy.hamming(framesamp)

    #Compute the stft
    X = scipy.array([np.array(scipy.fftpack.fft(w*x[i:i+framesamp], fs)) for i in range(0, len(x)-framesamp, hopsamp)])
    X = np.array( X[:,0:int(X.shape[1]/2)] )    #Take only the fisrt half part of the vector (only positive values in axis x)

    #Normalitzation of the input data to (0, 1)
    if norm:    
        z = np.abs(X).T   #shape: frequency x time 
        z = z - z.min(axis=0)
        z = z / z.max(axis=0)    
    else:
        z = np.abs(X).T

    return z    #shape: frequency x time. For now, only returns the magnitude.

def quantitzation(x, qLevels=20, Linear=True, Scale=False, show=True, max_value=None, min_value=None):
    xg = np.array(x)

    #print("max x: ", np.max(xg))
    #print("min x: ", np.min(xg))

    #Normalitzation of the input data to (0, 1)
    
    if (max_value == None) | (min_value == None):
        maxim = np.max(xg)
        minim = np.min(xg)
        xg = (xg-minim)/(maxim-minim)

    elif ((max_value != None) and (min_value != None)):
        xg = np.clip(xg, min_value, max_value)
        maxim = max_value
        minim = min_value
        xg = np.array((x-minim)/(maxim-minim))
        #print("max x: ", np.max(xg))
        #print("min x: ", np.min(xg))
    
    else:
        return -1, -1
            
    
    #Non linear quantitzier
    if not Linear:
        z = np.sqrt(xg)
        #z = np.log2(xg)
    else:
        z = xg
    
    #Liniar quantitzier
    q = 1/(qLevels-1)
    y = np.round(z/q)
    
    #Remove items that are quantitzied at the same level
    mask = -1
    if Scale:

        #Create the mask to reduce the input array size (it is suposed to be ordered)
        mask = np.zeros(x.size)
        where = np.flatnonzero
        pos = np.r_[0, where(~np.isclose(y[1:], y[:-1], equal_nan=True)) + 1]
        mask[pos] = 1

        #Apply mask to input xg and output y
        xq = np.ma.masked_where(mask!=1, xg)
        yq = np.ma.masked_where(mask!=1, y)/(qLevels-1)

        xq = np.ma.compressed(xq)
        yq = np.ma.compressed(yq)

    else:
        xq = xg
        yq = y

    if show:
        print("-"*10, "Quantitzation", "-"*10)
        #print("q: ", q)
        print("Input size: ", x.size)
        print("Quantitzation levels: ", int(1/q)+1)
        print("q: ", q)
        """
        plt.plot(xg, y/qLevels)
        plt.title("Non Linear quant. + linear")
        plt.xlabel("input x")
        plt.ylabel("output y, {} levels".format(qLevels))
        plt.show()
        """
        print("Linear: ", Linear)
        print("Output size: ", yq.size)
        print("-"*35)
    
    return yq, mask

def fft(x, fs, framesz):
    framesamp = int(framesz*fs)
    w = scipy.hamming(framesamp)
    i = 0
    X = scipy.array([scipy.fftpack.fft(w*x[:i+framesamp])])
    return X

def ntf_separation():
    pass