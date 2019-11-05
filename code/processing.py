import numpy as np
from scipy.fftpack import dct
from IPython import embed


"""
RESULTA QUE AL PAPER UTILITZEN EL MATEIX QUE VAM ESTUDIAR A CLASSE DELS MFCCS

FUNCIONS QUE ANIREM APLICANT DE FORMA CRONOLOGICA
PER FINALMENT CONSEGUIR ELS MFCCs:

    -- FRAMING
        Objectiu: Aconseguir senyal estacionari per aplicar fft
                En el paper utilitzen finestres de 40ms amb un
                overlapping del 50%
        Input: Signal de Fs*10s = 441000
        Output: Diferent frames on aplicar stft
    -- HAMMING
        Objectiu: Reduir 'spectral leakage'
        Input: frames
        Output: frames*hamming
    -- FFT and Power spectrum
        Objectiu: Computar periodograma
        Input: frames
        Output: periodograma per cada frame
    -- Filter Banks
        Objectiu: Aconseguir un espectre més semblant al que els humans
                percebem amb la oida. La escala 'mel' és més discriminativa
                en freq baixes i menys a freq altes (els filtres triangulars
                són més amplis). Al paper utilitzen 40 filtres i descarten els
                20 últims.
        Input: periodograma
        Ouput: vector de 40 elements on cada valor és l'energia que hi havia
                a cada filtre
    -- MFCCs
        Objectiu: Decorrelar els filter bank, ja que diuen que pot ser
                problematic. Realment nomes retornem una versió compresa dels
                filter banks.
        Input: energia dels filtres
        Ouput: MFFCs
    -- Delta / Acceleration coefficients (OPCIONAL)
        Objectiu: Calcular les trajectories dels coeficients MFCC.
                Diuen que els MFCC feature vector nomès ens dona info del
                envelope dun sol frame i que no tenim cap info dinàmica de com
                varien aquests MFCC. Normalment s'obté tants coeficients delta
                com coeficients MFCC, i s'appendejan al mateix vector. Al paper
                diuen que ho calculen utilitzan una finestra de 9 frames, i si
                és el mateix valor N que he vist per internet em sembla raro
                perque normalment és 2.
        Input: MFCCs
        Ouput: MFCCs + Acceleration coefficients
    -- Mean normalization
        Objectiu: Millorar la relació senyal soroll SNR.
        Input: coeficients
        Outpout: coeficients normalitzats
"""

def framing(signal):

    return frames

def hamming(frames):

    return frames

def fft(frames):

    return periodogram

def filterBanks(periodogram):

    return filter_banks

def MFCC(filter_banks):

    return mffcs

def deltaAcceleration(mfccs):

    return mfccs

def normalize(mfccs):

    return mfccs
