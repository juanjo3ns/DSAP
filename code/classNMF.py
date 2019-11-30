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

class NMF:
    def __init__(self):
        self.HAMMING_WINDOW = 0.040 #seconds
        self.SAMPLE_FREQ = 22050 #Hz
        self.NFFT = 2048
        self.OVERLAP = 0.5
        self.win_len = int(self.HAMMING_WINDOW * self.SAMPLE_FREQ)
        self.hop = int(self.OVERLAP * self.win_len)
        self.n_components = 20

    def sourceSeparation_components(self): # 4 COMPOMNENTS

        # Input Signal
        audio, sr = librosa.load('audio_task5/train/02_003127.wav')
        print("Input Sound:")
        ipd.display(ipd.Audio(audio, rate=sr))

        # Hamming Window: 60 ms + 25% overlap
        X = librosa.stft(audio, n_ftt=n_ftt, hop_len=OVERLAP * win_len, win_length=win_len, window='hamm')

        # Input Spectrum
        X_dB = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15,4))
        librosa.display.specshow(X_dB, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()

        # Magnitude spectogram
        X1, X1_phase = librosa.magphase(X)
        X1 = abs(X1)

        # Nonnegative Decompose
        W, H = librosa.decompose.decompose(X1, n_components=n_components, sort=True)
        print("Matrix W", W.shape)
        W_dB = librosa.amplitude_to_db(abs(W))
        plt.figure(figsize=(15,4))
        librosa.display.specshow(W_dB, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()
        H_dB = librosa.amplitude_to_db(abs(H))
        print("Matrix H", H.shape)
        plt.figure(figsize=(15,4))
        librosa.display.specshow(H_dB, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()

        # Reconstruct the complex spectrum
        # Display of spectral profiles
        plt.figure(figsize=(15,7))
        logW = np.log10(W)
        for n in range(n_components):
            plt.subplot(np.ceil(n_components/2.0), 2, n+1)
            plt.plot(logW[:,n])
            plt.ylim(-3, logW.max())
            plt.xlim(0, W.shape[0])
            plt.ylabel('Component %d' %n)

        #Display of temporal activations
        plt.figure(figsize=(14,7))
        for n in range(n_components):
            plt.subplot(np.ceil(n_components/2.0), 2, n+1)
            plt.plot(S[n])
            plt.ylim(0, H.max())
            plt.xlim(0, H.shape[1])
            plt.ylabel('Component %d' %n)

        # Recreation of the original components
        # This are the components that must be send to the RNN
        for n in range(n_components):
            Y1 = scipy.outer(W[:,n], H[n])*X1_phase # STFT of a single component
            iaudio = librosa.istft(Y1)

            if n == 0: out_0 = iaudio
            elif n == 1: out_1 = iaudio
            elif n == 2: out_2 = iaudio
            elif n == 3: out_3 = iaudio

            print('Component {}:'.format(n))
            ipd.display(ipd.Audio(iaudio, rate=sr))

        # Output spectrum
        Y1_dB = librosa.amplitude_to_db(Y1)
        plt.figure(figsize=(15,4))
        librosa.display.specshow(Y1, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()

        # Full Reconstruction
        Y = np.dot(W,H) * X1_phase
        Y_dB = librosa.amplitude_to_db(Y)
            # Spectrum
        plt.figure(figsize=(15,4))
        librosa.display.specshow(Y_dB, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()
            # Display
        print("Output:")
        re_audio = librosa.istft(Y, hop_length=hop, window='hamm', length=len(audio))
        ipd.display(ipd.Audio(re_audio, rate=sr))

        # Substract separated audio from the input audio

        # print("Cleaned Audio:")
        # clean_audio = audio - re_audio
        # ipd.display(ipd.Audio(clean_audio, rate=sr))

        # plt.plot(audio)
        # plt.plot(re_audio, 'r')

        # plt.plot(audio)
        # plt.plot(clean_audio, 'y')

        return out_0, out_1, out_2, out_3

    def sourceSeparation_activations(self): # ACTIVATIONS MATRIX

        # Input Signal
        audio, sr = librosa.load('/home/data/audio/audio_task5/train/02_003127.wav')
        print("Input Sound:")
        ipd.display(ipd.Audio(audio, rate=sr))

        # Hamming Window: 60 ms + 25% overlap
        X = librosa.stft(audio, n_fft=self.NFFT, hop_length=self.hop, win_length=self.win_len, window='hamm')

        # Input Spectrum
        X_dB = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(15,4))
        librosa.display.specshow(X_dB, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()

        # Magnitude spectogram
        X1, X1_phase = librosa.magphase(X)
        X1 = abs(X1)

        # Nonnegative Decompose
        W, H = librosa.decompose.decompose(X1, n_components=self.n_components, sort=True)
        print("Matrix W", W.shape)
        W_dB = librosa.amplitude_to_db(abs(W))
        plt.figure(figsize=(15,4))
        librosa.display.specshow(W_dB, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()
        H_dB = librosa.amplitude_to_db(abs(H))
        print("Matrix H", H.shape)
        plt.figure(figsize=(15,4))
        librosa.display.specshow(H_dB, sr=sr, x_axis ='time', y_axis='log')
        plt.colorbar()

        return H # Activation Matrix
