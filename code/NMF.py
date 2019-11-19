import librosa, librosa.display
import scipy, pylab
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
import mongodb_api as mongo
#from __future__ import division
from numpy import fft 
from IPython import embed
import IPython.display as ipd 
import wavio

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

def nmf_separation_1(): 
    # NMF is an algorithm that factorizes a nonnegative matrix into two nonnegative matrices
    # It is an unsupervised iterative algorithm that minimizes the distance between the original matrix and the product of the decomposed 
    audio, sr = librosa.load('./audio-dev/train/02_002227.wav') 
    ipd.Audio(audio, rate=sr)
    S = stft(audio)    # Librosa també conté la funció stft
    S_dB = librosa.amplitude_to_db(abs(S))
    plt.figure(figsize=(14,4))
    librosa.display.specshow(S_dB, sr=sr, x_axis ='time', y_axis='log')
    plt.colorbar()
    X, X_phase = librosa.magphase(S)
    n_components = 8 # Number of components we want to separate our audio into 
    W, H = librosa.decompose.decompose(X, n_components=n_components, sort=True)
    print(X.shape)
    print(W.shape)
    print(H.shape)

    # Display of spectral profiles 
    plt.figure(figsize=(13,7))
    logW = np.log10(W)
    for n in range(n_components): 
        plt.subplot(np.ceil(n_components/2.0, 2, n+1))
        plt.plot(logW[:,n])
        plt.ylim(-3, logW.max())
        plt.xlim(0, W.shape[0])
        plt.ylabel('Component %d' %n)
    
    # Display of temporal activations
    plt.figure(figsize=(13,7))
    for n in range(n_components):
        plt.subplot(np.ceil(n_components/2.0, 2, n+1))
        plt.plot(H[n])
        plt.ylim(0, H.max())
        plt.xlim(0, H.shape[1])
        plt.ylabel('Component %d' %n)

    # Recreation of the original components
    for n in range(n_components):
        Y = scipy.outer(W[:,n], H[n])*X_phase # STFT of a single component
        iaudio = librosa.istft(Y)

        print('Component {}:'.format(n))
        ipd.display(ipd.Audio(iaudio, rate=sr))
    
    # Full-mix reconstruction -- això no ens cal, però és per provar-ho ara
    Y = np.dot(W,H) * X_phase
    re_audio = librosa.istft(Y, length=len(audio))
    ipd.Audio(re_audio, rate=sr)

    # Residual Sound 
    res = audio - re_audio
    res[0] = 1
    ipd.Audio(res, rate=sr)
    
# nmf_separation()


def nmf_separation_2(): #NO COMPLET 
    sample, _ librosa.load('./audio-dev/train/02_002227.wav') 
    ipd.Audio(sample, rate=sr)
    X = librosa.stft(sample)
    X_mag, X_phase = librosa.magphase(X)
    X_dB = librosa.amplitude_to_db(X_mag)
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(X2_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()  

    # NMF - update on H, not on W
    W = librosa.util.normalize(X1_mag, norm=2, axis=0)
    WTX = W.T.dot(X2_mag)
    WTW = W.T.dot(W)
    H = np.random.ranf(X.shape[1])
    eps = 0.01 
    for _ in range(100): 
        H = H*(WTX + eps)/(WTW.dot(H) + eps)
    H.shape
    plt.imshow(H.T.dot(H))

