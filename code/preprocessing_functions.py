from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def spectrogram(x, fs, show=False):
    #fs = 44100
    print(x.shape)
    f, t, Sxx = signal.spectrogram(x, fs)
     
    if show:
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
    return Sxx


