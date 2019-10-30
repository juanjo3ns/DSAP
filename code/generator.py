import numpy as np 
import scipy as sc
from scipy import io 
from scipy.io import wavfile
import pylab
from IPython import embed
import matplotlib.pyplot as plt
import cv2 as cv2
#import wavio


import dataset as dataset
import preprocessing_functions as pf

import torch
from torch.utils.data import TensorDataset, DataLoader

class generator():
    def __init__(self):
        self.data = DataLoader(dataset=dataset.WAV_dataset(mode="train"), batch_size=1, shuffle=False)
        #self.spectrogram()
        self.sftp_run(1)

    def spectrogram(self):
        for d in self.data:
            #d[0] = "airport-barcelona-0-0-a.wav"
            #wav = scipy.io.wavfile.read("/home/data/audio/" + str(d[0]), mmap=False)
            
            
            wav = wavio.read("/home/data/audio/" + str(d[0][0]))
            rate = wav.rate
            wav = wav.data[:,1]
            
            #Sxx = pf.spectrogram(wav, rate, show=True)
            #Sxx = pf.spect(wav)
            #self.save(Sxx, d[0].split(".")[0])
            pf.spectrogram(wav, rate, name=d[0][0].split(".")[0])
            
            print(values)
        return "done"
    
    def sftp_run(self, x, t_size=None, f_size=None, fs=None, framesz=None, show=True):
        f0 = 500      # Compute the STFT of a 440 Hz sinusoid
        f1 = 3000
        fs = 8000        # sampled at 8 kHz
        T = 12            # lasting 5 seconds
        framesz = 0.005  # with a frame size of 50 milliseconds
        hop = framesz     # and hop size of 20 milliseconds.

        # Create test signal and STFT.
        t = sc.linspace(0, T, T*fs, endpoint=False)
        
        
        x = 20*sc.sin(2*sc.pi*f0*t) + sc.sin(2*sc.pi*f1*t) + 1/2*sc.sin(2*sc.pi*f1/2*t)
        X = np.array(pf.stft(x, fs, framesz, hop, norm=False))

        max_Spect = np.max(X)

        f = np.linspace(0,int(fs/2), X.shape[0])
        yq, mask = pf.quantitzation(f, qLevels=255,Linear=True, Scale=True, show=False)


        h = np.zeros((X.shape[1], 255))

        
        for i in range(0, X.shape[1]):
            z = X[:, i]

            zq = np.ma.masked_where(mask!=1, z)
            zq = np.ma.compressed(zq)

            h[i], _ = pf.quantitzation(zq, qLevels=255, Linear=True, Scale=False, max_value=max_Spect, min_value=0, show=False)

            #plt.plot(yq, h[i])
            #plt.show()
            #break

        self.save(h, "pol")
        
        s = cv2.imread("/home/data/spectros/pol.png", 0)

        if show:
        
            print(h.shape)
            pylab.figure()
            
            pylab.imshow(sc.absolute(s.T), origin='lower', aspect='auto', interpolation='nearest')
            pylab.xlabel('Time')
            pylab.ylabel('Frequency')
            pylab.show()
            maxium = np.max(X)
        
        


    def sftp(self):
        f0 = 500      # Compute the STFT of a 440 Hz sinusoid
        f1 = 3000
        fs = 8000        # sampled at 8 kHz
        T = 12            # lasting 5 seconds
        framesz = 0.005  # with a frame size of 50 milliseconds
        hop = framesz     # and hop size of 20 milliseconds.

        # Create test signal and STFT.
        t = sc.linspace(0, T, T*fs, endpoint=False)
        

        x = 20*sc.sin(2*sc.pi*f0*t) + sc.sin(2*sc.pi*f1*t) + 1/2*sc.sin(2*sc.pi*f1/2*t)
        X = np.array(pf.stft(x, fs, framesz, hop, norm=False))
 
        
        z = X[:, 0]
        f = np.linspace(0,int(fs/2), z.size)
        yq, mask = pf.quantitzation(f, qLevels=255,Linear=True, Scale=True, show=False)
        

        zq = np.ma.masked_where(mask!=1, z)
        zq = np.ma.compressed(zq)

        print(zq.shape)
        print(np.max(zq))
        zq, mask = pf.quantitzation(zq, qLevels=255, Linear=True, Scale=False, max_value=1000, show=False)
        print(zq.shape)
        print(np.max(zq))


        plt.plot(yq, zq)
        plt.show()
       

        pylab.figure()
        
        pylab.imshow(sc.absolute(X), origin='lower', aspect='auto',
                    interpolation='nearest')
        
        #pylab.imshow(b)
        pylab.xlabel('Time')
        pylab.ylabel('Frequency')
        pylab.show()
        

        return None


    def save(self, img, name):
        path_to_save = "/home/data/spectros/"
        print(path_to_save + name + ".png")
        cv2.imwrite(path_to_save + name + ".png", img)
        
        


if __name__ == '__main__':
	ds = generator()
