import scipy, pylab
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import mongodb_api as mongo
#from __future__ import division
from numpy import fft 
#from pytfd import helpers as h 

def spectrogram(audio, sf, name):
    fmin = 100
    fmax = 20000
    
    if sf < fmin:
        print('Sampling frequency too low.')
        sys.exit(1)

    # convert to mono
    #sig = np.mean(audio, axis=1)
    sig = audio
    # vertical resolution (frequency)
    # number of points per segment; more points = better frequency resolution
    # if equal to sf, then frequency resolution is 1 Hz
    npts = int(sf/16)
    npts = int(sf/4)

    # horizontal resolution (time)
    # fudge factor to keep the number of frequency samples close to 1000
    # (assuming an image width of about 1000 px)
    # negative values ought to be fine
    # this needs to change if image size becomes parametrized
    winfudge = 1 - ((np.shape(sig)[0] / sf) / 1000)

    print('Calculating FFT...')
    f, t, Sxx = signal.spectrogram(sig, sf, nperseg=npts, noverlap=int(winfudge * npts))

    # FFT at high resolution makes way too many frequencies
    # set some lower number of frequencies we would like to keep
    # final result will be even smaller after pruning
    nf = 1000
    # generate an exponential distribution of frequencies
    # (as opposed to the linear distribution from FFT)
    b = fmin - 1
    a = np.log10(fmax - fmin + 1) / (nf - 1)
    freqs = np.empty(nf, int)
    for i in range(nf):
        freqs[i] = np.power(10, a * i) + b
    # list of frequencies, exponentially distributed:
    freqs = np.unique(freqs)

    # delete frequencies lower than fmin
    fnew = f[f >= fmin]
    cropsize = f.size - fnew.size
    f = fnew
    Sxx = np.delete(Sxx, np.s_[0:cropsize], axis=0)

    # delete frequencies higher than fmax
    fnew = f[f <= fmax]
    cropsize = f.size - fnew.size
    f = fnew
    Sxx = Sxx[:-cropsize, :]

    findex = []
    # find FFT frequencies closest to calculated exponential frequency distribution
    for i in range(freqs.size):
        f_ind = (np.abs(f - freqs[i])).argmin()
        findex.append(f_ind)

    # keep only frequencies closest to exponential distribution
    # this is usually a massive cropping of the initial FFT data
    fnew = []
    for i in findex:
        fnew.append(f[i])
    f = np.asarray(fnew)
    Sxxnew = Sxx[findex, :]
    Sxx = Sxxnew

    print('Generating the image...')
    plt.pcolormesh(t, f, np.log10(Sxx))
    plt.ylabel('f [Hz]')
    plt.xlabel('t [sec]')
    plt.yscale('symlog')
    #plt.ylim(fmin, fmax)

    # TODO: make this depend on fmin / fmax
    # right now I'm assuming a range close to 10 - 20000
    yt = np.arange(10, 100, 10)
    yt = np.concatenate((yt, 10 * yt, 100 * yt, 1000 * yt))
    yt = yt[yt <= fmax]
    yt = yt.tolist()
    #plt.yticks(yt)
    print(Sxx.shape)
    print(Sxx)
    print(f)
    #plt.show()
    path_to_save = "/home/data/spectrogram/"
    plt.savefig(path_to_save + "plot.png", dpi=300, bbox_inches='tight')
    img = cv2.imread(path_to_save + "plot.png")
    img = img[35:1125, 195:1675]
    path_to_save = "/home/data/spect/"
    cv2.imwrite(path_to_save + name +".png", img)

def stft_deprecated(x,w, L=None):
    #Short-time Fourier Transform 
    #bib: https://gist.github.com/endolith/2784026 
    # L is the overlap, see http://cnx.org/content/m10570/latest/
    N = len(x)
    #T = len(w)
    if L is None:
        L = N
    # Zerro pad the window
    w = h.zeropad(w, N)
    X_stft = []
    points = range(0, N, N//L)
    for i in points:
        x_subset = h.subset(x, i, N)
        fft_subset = fft(x_subset * w)
        X_stft.append(fft_subset)
    X_stft = array(X_stft).transpose()
    return X_stft 

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

def quantitzation(x, norm=True, qLevels=20, Linear=True, Scale=False,show=True):
    x = np.array(x)

    #Normalitzation of the input data to (0, 1)
    if norm or True:
        maxim = np.max(x)
        minim = np.min(x)
        xg = (x-minim)/(maxim-minim)
    
    #Non linear quantitzier
    if not Linear:
        z = np.sqrt(xg)
        #z = np.log2(xg)
    else:
        z = xg
    
    #Liniar quantitzier
    q = 1/qLevels
    y = np.round(z/q)

    #Remove items that are quantitzied at the same level
    mask = [0]
    if Scale:
        #Create the mask to reduce the input array size
        mask = [0]
        for i in range(1, y.size):
            if y[i] != y[i-1]:
                mask.append(1)
            else:
                mask.append(0)
        mask = np.array(mask)

        #Apply mask to input xg and output y
        xq = np.ma.masked_where(mask!=1, xg)
        yq = np.ma.masked_where(mask!=1, y)/qLevels

        xq = np.ma.compressed(xq)
        yq = np.ma.compressed(yq)

    else:
        xq = xg
        yq = y

    if show:
        print("-"*10, "Quantitzation", "-"*10)
        #print("q: ", q)
        print("Input size: ", x.size)
        print("Quantitzation levels: ", int(1/q))
        print("q: ", q)
        plt.plot(xg, y/qLevels)
        plt.title("Non Linear quant. + linear")
        plt.xlabel("input x")
        plt.ylabel("output y, {} levels".format(qLevels))
        plt.show()
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