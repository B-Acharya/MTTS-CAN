import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks, stft
import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks, stft

"""
This module contains all the class and the methods for transforming 
a BVP signal in a BPM signal.
Adapted code from pyVHR
"""
def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes RMSE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c, j], 2)

    # -- final RMSE
    RMSE = np.sqrt(df/m)
    return RMSE


def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAE = df/m
    return MAE

def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None):
    n, m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = np.zeros((n, m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            diff[c, j] = bpmGT[i]-bpmES[c, j]
    return diff

class BVPsignal:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    #nFFT = 2048  # freq. resolution for STFTs
    step = 1       # step in seconds

    def __init__(self, data, fs, startTime=0, minHz=0.75, maxHz=4., verb=False):
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fs = fs                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz
        nyquistF = self.fs/2
        fRes = 0.5
        self.nFFT = max(2048, (60*2*nyquistF) / fRes)

    def spectrogram(self, winsize=5):
        """
        Compute the BVP signal spectrogram restricted to the
        band 42-240 BPM by using winsize (in sec) samples.
        """

        # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=self.fs*winsize,
                       noverlap=self.fs*(winsize-self.step),
                       boundary='even',
                       nfft=self.nFFT)
        Z = np.squeeze(Z, axis=0)

        # -- freq subband (0.75 Hz - 4.0 Hz)
        minHz = 0.65
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        self.spect = np.abs(Z[band, :])     # spectrum magnitude
        self.freqs = 60*F[band]            # spectrum freq in bpm
        self.times = T                     # spectrum times

        # -- BPM estimate by spectrum
        self.bpm = self.freqs[np.argmax(self.spect, axis=0)]

    def displaySpectrum(self, display=False, dims=3):
        """Show the spectrogram of the BVP signal"""

        # -- check if bpm exists
        try:
            bpm = self.bpm
        except AttributeError:
            self.spectrogram()
            bpm = self.bpm

        t = self.times
        f = self.freqs
        S = self.spect

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=S, x=t, y=f, colorscale="viridis"))
        fig.add_trace(go.Scatter(
            x=t, y=bpm, name='Frequency Domain', line=dict(color='red', width=2)))

        fig.update_layout(autosize=False, height=420, showlegend=True,
                          title='Spectrogram of the BVP signal',
                          xaxis_title='Time (sec)',
                          yaxis_title='BPM (60*Hz)',
                          legend=dict(
                              x=0,
                              y=1,
                              traceorder="normal",
                              font=dict(
                                family="sans-serif",
                                size=12,
                                color="black"),
                              bgcolor="LightSteelBlue",
                              bordercolor="Black",
                              borderwidth=2)
                          )

        fig.show(renderer=VisualizeParams.renderer)

    def getBPM(self, winsize=5):
        """
        Get the BPM signal extracted from the ground truth BVP signal.
        """
        self.spectrogram(winsize)
        return self.bpm, self.times
