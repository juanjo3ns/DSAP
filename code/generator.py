import numpy as np 
import scipy as sc
from scipy import io 
from scipy.io import wavfile
import pylab
from IPython import embed
import matplotlib.pyplot as plt
#import wavio


import dataset as dataset
import preprocessing_functions as pf

import torch
from torch.utils.data import TensorDataset, DataLoader

class generator():
    def __init__(self):
        self.data = DataLoader(dataset=dataset.WAV_dataset(mode="train"), batch_size=1, shuffle=False)
        #self.spectrogram()
        self.sftp()


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
            
            
        return "done"
    
    def sftp(self):
        f0 = 500      # Compute the STFT of a 440 Hz sinusoid
        f1 = 3000
        fs = 8000        # sampled at 8 kHz
        T = 12            # lasting 5 seconds
        framesz = 0.005  # with a frame size of 50 milliseconds
        hop = framesz     # and hop size of 20 milliseconds.

        # Create test signal and STFT.
        t = sc.linspace(0, T, T*fs, endpoint=False)
        

        x = sc.sin(2*sc.pi*f0*t) + sc.sin(2*sc.pi*f1*t)
        X = np.array(pf.stft(x, fs, framesz, hop))
 
        print("X saphe: ", X.shape)
        print("first column", X[1,:].shape)
        print("min: ", np.min(X[1,:]))
        print("max: ", np.max(X[1,:]))
        
        z = X[:, 1]
        f = np.linspace(0,int(fs/2), z.size)
        a = pf.quantitzation(f)
        plt.plot(a, z)
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
        path_to_save = "/home/data/spectrogram/"
        print(path_to_save + name + ".png")
        img.savefig(path_to_save + 'foo.png')
        
        #cv2.imwrite(path_to_save + name, img)


if __name__ == '__main__':
	ds = generator()
