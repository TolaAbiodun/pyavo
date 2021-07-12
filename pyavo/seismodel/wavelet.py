# -*- coding: utf-8 -*-
"""
Functions to generate a ricker and bandpass wavelet
created by: Tola Abiodun
create date: 01/06/2021
"""

import numpy as np
import scipy.signal as signal


def plot_ricker(sample_rate=0.001, length=0.512, c_freq=25):
    """
    Generate a zero-phase wavelet plot.

    :param sample_rate: sample rate in seconds (float, int)
    :param length: length of time (dt) in seconds (float, int)
    :param c_freq: central frequency of wavelet (cycles/seconds or Hz). (float, int)
    :return: ndarray

    Example:
    -------
    plot_ricker(sample_rate,duration,c_freq)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    t_min = -length / 2
    t_max = (length - sample_rate) / 2
    wv_time = np.linspace(t_min, t_max, int(length / sample_rate))
    wv_amp = (1. - 2. * (np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2)) * np.exp(
        -(np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2))
    plt.figure(figsize=(7, 4))
    plt.plot(wv_time, wv_amp, lw=2, color='black', alpha=0.5)
    plt.fill_between(wv_time, wv_amp, 0, wv_amp > 0.0, interpolate=False, color='blue', alpha=0.5)
    plt.fill_between(wv_time, wv_amp, 0, wv_amp < 0.0, interpolate=False, color='red', alpha=0.5)
    plt.title('%d Hz Ricker wavelet' % c_freq, fontsize=16)
    plt.xlabel('TWT (s)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.ylim((-1.1, 1.1))
    plt.xlim((min(wv_time), max(wv_time)))
    plt.grid()
    plt.show()


def ricker(sample_rate=0.001, length=0.512, c_freq=25):
    """
    Generate time and amplitude values for a zero-phase wavelet.

    The second derivative of the Gaussian function or the third derivative of the normal-probability density function.

    A Ricker wavelet is often used as a zero-phase embedded wavelet in modeling and synthetic seismogram manufacture. Norman H. Ricker (1896–1980), American geophysicist.

    :param sample_rate: sample rate in seconds (float, int)
    :param length: length of time (dt) in seconds (float, int)
    :param c_freq: central frequency of wavelet (cycles/seconds or Hz). (float, int)
    :return: ndarray

    Example:
    -------
    time, wavelet = (sample_rate,duration,c_freq)
    """
    import numpy as np

    t_min = -length / 2
    t_max = (length - sample_rate) / 2
    wv_time = np.linspace(t_min, t_max, int(length / sample_rate))
    wv_amp = (1. - 2. * (np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2)) * np.exp(
        -(np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2))

    return wv_time, wv_amp


def bandpass(f1, f2, f3, f4, phase, dt, wvlt_length):
    """
    Calculate a trapezoidal bandpass wavelet

    f1: Low truncation frequency of wavelet in Hz
    f2: Low cut frequency of wavelet in Hz
    f3: High cut frequency of wavelet in Hz
    f4: High truncation frequency of wavelet in Hz
    phase: wavelet phase in degrees
    dt: sample rate in seconds
    wvlt_length: length of wavelet in seconds

    Usage:
        t, wvlt = wvlt_ricker(f1, f2, f3, f4, phase, dt, wvlt_length)

    Reference
        Wes Hamlyn, 2014.
    """

    from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift

    nsamp = int(wvlt_length / dt + 1)

    freq = fftfreq(nsamp, dt)
    freq = fftshift(freq)
    aspec = freq * 0.0
    pspec = freq * 0.0

    # Calculate slope and y-int for low frequency ramp
    M1 = 1 / (f2 - f1)
    b1 = -M1 * f1

    # Calculate slop and y-int for high frequency ramp
    M2 = -1 / (f4 - f3)
    b2 = -M2 * f4

    # Build initial frequency and filter arrays
    freq = fftfreq(nsamp, dt)
    freq = fftshift(freq)
    filt = np.zeros(nsamp)

    # Build LF ramp
    idx = np.nonzero((np.abs(freq) >= f1) & (np.abs(freq) < f2))
    filt[idx] = M1 * np.abs(freq)[idx] + b1

    # Build central filter flat
    idx = np.nonzero((np.abs(freq) >= f2) & (np.abs(freq) <= f3))
    filt[idx] = 1.0

    # Build HF ramp
    idx = np.nonzero((np.abs(freq) > f3) & (np.abs(freq) <= f4))
    filt[idx] = M2 * np.abs(freq)[idx] + b2

    # Unshift the frequencies and convert filter to fourier coefficients
    filt2 = ifftshift(filt)
    Af = filt2 * np.exp(np.zeros(filt2.shape) * 1j)

    # Convert filter to time-domain wavelet
    wvlt = fftshift(ifft(Af))
    wvlt = np.real(wvlt)
    wvlt = wvlt / np.max(np.abs(wvlt))  # normalize wavelet by peak amplitude

    # Generate array of wavelet times
    t = np.linspace(-wvlt_length * 0.5, wvlt_length * 0.5, nsamp)

    # Apply phase rotation if desired
    if phase != 0:
        phase = phase * np.pi / 180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase) * wvlt - np.sin(phase) * wvlth

    return t, wvlt



