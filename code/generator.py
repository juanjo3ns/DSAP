import numpy as np 
import scipy
from scipy import io 
from scipy.io import wavfile
import wavio


import dataset as dataset
import preprocessing_functions as pf

import torch
from torch.utils.data import TensorDataset, DataLoader

class generator():
    def __init__(self):
        self.data = DataLoader(dataset=dataset.WAV_dataset(mode="train"), batch_size=1, shuffle=False)
        self.spectrogram()


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


    def save(self, img, name):
        path_to_save = "/home/data/spectrogram/"
        print(path_to_save + name + ".png")
        img.savefig(path_to_save + 'foo.png')
        
        #cv2.imwrite(path_to_save + name, img)


if __name__ == '__main__':
	ds = generator()
