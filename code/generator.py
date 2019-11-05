import numpy as np
import scipy as sc
from scipy import io
from scipy.io import wavfile
import pylab
from IPython import embed
import matplotlib.pyplot as plt
import cv2 as cv2
import wavio
from IPython import embed


import dataset as dataset
import preprocessing_functions as pf

import torch
from torch.utils.data import TensorDataset, DataLoader

class generator():
    def __init__(self):
        self.data = DataLoader(dataset=dataset.WAV_dataset(mode="train"), batch_size=1, shuffle=False)
        self.spectrogram()
        #self.stft(0,0)


    def test(self):
        x, fs = self.sinus_test()
        wavio.write("/home/data/spectros/test.wav", x, fs, sampwidth=3)

    def sinus_test(self):
        f0 = 500      # Compute the STFT of a 440 Hz sinusoid
        f1 = 3000
        fs = 8000        # sampled at 8 kHz
        T = 12            # lasting 5 seconds
        framesz = 0.005  # with a frame size of 50 milliseconds
        hop = framesz     # and hop size of 20 milliseconds.

        # Create test signal and STFT.
        t = sc.linspace(0, T, T*fs, endpoint=False)

        x = 20*sc.sin(2*sc.pi*f0*t) + sc.sin(2*sc.pi*f1*t) + 1/2*sc.sin(2*sc.pi*f1/2*t)

        return x, fs

    def spectrogram(self):
        for i, d in enumerate(self.data):
            #d[0] = "airport-barcelona-0-0-a.wav"

            #Read audio file
            wav = wavio.read("/home/data/audio/" + str(d[0][0]))
            embed()
            rate = wav.rate
            wav = wav.data[:,0]

            print(np.max(wav))
            print(wav)

            Sxx = self.stft(wav, fs=rate, framesz=0.005, show=True, depth=255)
            print(Sxx.shape)
            plt.plot(Sxx[76])
            plt.show()

            if i == 0:
                break
        return "done"

    def stft(self, x, fs, t_size=None, f_size=None, framesz=0.005,hop=0.005, depth=255,show=False):

        hop = framesz
        #Compute the stft:
        X = np.array(pf.stft(x, fs, framesz, hop, norm=False))

        #Maximum to normalize
        max_Spect = np.max(X)


        #Quantitzation of the spectrogram
        f = np.linspace(0,int(fs/2), X.shape[0])
        yq, mask = pf.quantitzation(f, qLevels=depth,Linear=True, Scale=True, show=False)

        Sxx = np.zeros((X.shape[1], depth))
        for i in range(0, X.shape[1]):
            z = X[:, i]

            zq = np.ma.masked_where(mask!=1, z)
            zq = np.ma.compressed(zq)

            Sxx[i], _ = pf.quantitzation(zq, qLevels=depth, Linear=True, Scale=False, max_value=max_Spect, min_value=0, show=False)

            #plt.plot(yq, h[i])
            #plt.show()
            #break

        if show:

            print(Sxx.shape)
            pylab.figure()
            pylab.imshow(sc.absolute(Sxx.T), origin='lower', aspect='auto', interpolation='nearest')
            pylab.xlabel('Time')
            pylab.ylabel('Frequency')
            pylab.show()
            maxium = np.max(X)

        return Sxx
        #self.save(h, "pol")
        #s = cv2.imread("/home/data/spectros/pol.png", 0)

    def save(self, img, name):
        path_to_save = "/home/data/spectros/"
        print(path_to_save + name + ".png")
        cv2.imwrite(path_to_save + name + ".png", img)




if __name__ == '__main__':
	ds = generator()
