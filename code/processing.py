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
class Processing:
    def __init__(self):
        self.FRAME_SIZE = 0.04
        self.FRAME_STRIDE = 0.02
        self.NFFT = 512
        self.N_FILT = 40
        self.NUM_CEPS = 20
        self.DELTA_WINDOW = 9
        self.LENGTH_MAX = 441000

        self.sample_rate = None
        self.frame_length = None
        self.frame_step = None

        self.frames = None
        self.periodogram = None
        self.filter_banks = None
        self.mfccs = None

    def process(self, signal):
        self.framing(signal)
        self.hamming()
        self.fft()
        self.filterBanks()
        self.MFCC()
        # self.deltaAcceleration()
        self.normalize()
        self.scale()
        return self.mfccs, self.filter_banks, self.periodogram

    def framing(self, signal):
        self.sample_rate = signal.rate

        self.frame_length = self.FRAME_SIZE * self.sample_rate
        self.frame_step = self.FRAME_STRIDE * self.sample_rate
        self.frame_length = int(round(self.frame_length))
        self.frame_step = int(round(self.frame_step))

        signal_length = min(signal.data.size, self.LENGTH_MAX)
        signal.data = signal.data[:signal_length]
        num_frames = int(np.ceil(signal_length / self.frame_step))
        if num_frames > 500:
            embed()
        pad_signal_length = num_frames * self.frame_step + self.frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal.data, z)
        indices = np.tile(np.arange(0, self.frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        self.frames = pad_signal[indices.astype(np.int32, copy=False)]

    def hamming(self):
        self.frames *= np.hamming(self.frame_length)

    def fft(self):
        magnitude_frames = np.absolute(np.fft.rfft(self.frames, self.NFFT))
        self.periodogram = ((1.0 / self.NFFT) * ((magnitude_frames) ** 2))

    def filterBanks(self):
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_FILT + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((self.NFFT + 1) * hz_points / self.sample_rate)

        fbank = np.zeros((self.N_FILT, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, self.N_FILT + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        self.filter_banks = np.dot(self.periodogram, fbank.T)
        self.filter_banks = np.where(self.filter_banks == 0, np.finfo(float).eps, self.filter_banks)
        self.filter_banks = 20 * np.log10(self.filter_banks)

    def MFCC(self):
        self.mfccs = dct(self.filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (self.NUM_CEPS + 1)]

    def deltaAcceleration(self):
        deltas = np.zeros_like(self.mfccs)
        for t in range(self.mfccs[:,0].size):
            numerator = np.zeros(self.mfccs[0].size)
            denominator = np.zeros(self.mfccs[0].size)
            for n in range(1,self.DELTA_WINDOW+1):
                coef1 = np.zeros(self.mfccs[0].size)
                coef2 = np.zeros(self.mfccs[0].size)
                if t+n < self.mfccs[:,0].size:
                    coef1 = self.mfccs[t+n]
                if t-n >= 0:
                    coef2 = self.mfccs[t-n]
                numerator += n*(coef1 - coef2)
                denominator += n^2
            deltas[t] = numerator / 2*denominator
        self.mfccs = np.concatenate((self.mfccs, deltas), axis=1)

    def normalize(self):
        self.filter_banks -= (np.mean(self.filter_banks, axis=0) + 1e-8)
        self.mfccs -= (np.mean(self.mfccs, axis=0) + 1e-8)

    def scale(self):
        max = np.max(self.mfccs)
        min = np.min(self.mfccs)
        self.mfccs = 255*(self.mfccs - min)/(max-min)
