from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import mongodb_api as mongo


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



