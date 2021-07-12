# -*- coding: utf-8 -*-
"""
Functions to generate a three layer wedge model using the reflection coefficients and travel times
Original Script by Wes Hamlyn, 2014
Refactored by Tola Abiodun, 2021.
"""

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import gridspec


def get_rc(Vp, rho):
    """
    Calculates the Reflection Coefficient at the interface separating two layers. The ratio of amplitude
    of the reflected wave to the incident wave, or how much energy is reflected. Typical values of R
    are approximately −1 from water to air, meaning that nearly 100% of the energy is reflected and
    none is transmitted; ~0.5 from water to rock; and ~0.2 for shale to sand.

    :param Vp: P-wave velocity (m/s)
    :param rho: Layer density (g/cc)
    :return:
        rc_int: Reflection Coefficient
    """
    rc_int = []
    n_int = len(Vp) - 1

    for interval in range(0, n_int):
        z1 = Vp[interval] * rho[interval]
        z2 = Vp[interval + 1] * rho[interval + 1]
        rc = (z2 - z1) / (z1 + z2)
        rc = round(rc, 2)
        rc_int.append(rc)
    return rc_int


def calc_times(z_int: list, vp: list) -> list:
    """
    Calculate the travel time to a reflector.

    :param z_int: Depth to a reflector
    :param vp: P-Wave Velocity
    :return:
        t_int: Interface times

    Usage
        t_int = calc_times(z_int, vp)
    """
    nlayers = len(vp)
    nint = nlayers - 1

    t_int = []
    for i in range(0, nint):
        if i == 0:
            tbuf = z_int[i] / vp[i]
            t_int.append(tbuf)
        else:
            zdiff = z_int[i] - z_int[i - 1]
            tbuf = 2 * zdiff / vp[i] + t_int[i - 1]
            t_int.append(tbuf)

    return t_int


def time_samples(t_min: float, t_max: float, dt=0.0001) -> list:
    """
    Create regularly sampled time series defining model sampling.

    :param t_min: Minimum time duration
    :param t_max: Maximum time duration
    :param dt: Change in time, default = 0.0001
    :return: time
    """
    n_samp = int((t_max - t_min) / dt) + 1
    time = []
    for t in range(0, n_samp):
        time.append(t * dt)
    return time


def mod_digitize(rc_int: list, t_int: list, t: list) -> list:
    """
    Digitize a 3 layer model using reflection coefficients and interface times.

    :param rc_int: reflection coefficients corresponding to interface times
    :param t_int: interface times
    :param t: regularly sampled time series defining model sampling
    :return:
        rc: Reflection Coefficient

    Usage
        rc = digitize_model(rc_int, t_int, t)
    """

    import numpy as np
    n_int = len(rc_int) - 1
    n_samp = len(t)

    rc = list(np.zeros(n_samp, dtype='float'))
    layer = 0

    for i in range(0, n_samp):
        if t[i] >= t_int[layer]:
            rc[i] = rc_int[layer]
            layer += 1

        if layer > n_int:
            break
    return rc


def syn_seis(ref_coef: list, wav_amp):
    """
    Generate synthetic seismogram from convolved reflectivities and wavelet.

    :param ref_coef: Reflection coefficient
    :param wav_amp: wavelet amplitude
    :return:
        smg: array
            Synthetic seismogram
    """
    smg = np.convolve(ref_coef, wav_amp, mode='same')
    smg = list(smg)
    return smg


def int_depth(h_int: list, dh_min: float, dh_step: float, mod: int):
    """
    Computes the depth to an interface.

    :param h_int: depth to first interface
    :param dh_min: minimum thickness of layer 2
    :param dh_step: Thickness step from trace-to-trace (usually 1.0m)
    :param mod: model traces generated within a thickness or interval
    :return:
        d_inteface: depth to interface
    """
    d_interface = h_int
    d_interface.append(d_interface[0] + dh_min + dh_step * mod)
    return d_interface


def n_model(h_min: float, h_max: float, h_step=1):
    """
    Computes number of traces within an interval or thickness.

    :param h_min: minimum thickness
    :param h_max: maximum thickness
    :param h_step: thickness steps, default is 1
    :return:
        n_trace: number of traces
    """
    n_trace = int((h_max - h_min) / h_step + 1)
    return n_trace


def ricker(sample_rate: float, length: float, c_freq: float):
    """
    Generate time and amplitude values for a zero-phase wavelet. The second derivative of the Gaussian
    function or the third derivative of the normal-probability density function.

    :param sample_rate: sample rate in seconds
    :param length: length of time (dt) in seconds
    :param c_freq: central frequency of wavelet (cycles/seconds or Hz).
    :return:
        wv_time: zero-phase wavelet time
        wv_amp: zero-phase wavelet amplitude

    Usage:
        time, wavelet = (sample_rate,duration,c_freq)

    Reference:
        A Ricker wavelet is often used as a zero-phase embedded wavelet in modeling and synthetic
        seismogram manufacture. Norman H. Ricker (1896–1980), American geophysicist.
    """

    t_min = -length / 2
    t_max = (length - sample_rate) / 2
    wv_time = np.linspace(t_min, t_max, int(length / sample_rate))
    wv_amp = (1. - 2. * (np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2)) * np.exp(
        -(np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2))

    return wv_time, wv_amp


# Calculate the tuning trace and tuning thickness
def _tuning_trace(syn_zo):
    """
    Computes the tuning trace and thickness in a synthetic trace gather.

    :param syn_zo: Synthetic seismogram
    :param step: Trace steps
    :return:
        t_trace: tuning trace
        t_thick: tuning thickness
    """
    t_trace = np.argmax(np.abs(syn_zo.T)) % syn_zo.T.shape[1]
    return t_trace


def _tuning_thickness(syn_zo, step=1):
    """
    Computes the tuning thickness in a synthetic trace gather

    :param syn_zo: Synthetic seismogram
    :param step: Trace steps
    :return:
        t_thick: tuning thickness
    """
    t_thick = _tuning_trace(syn_zo) * step
    return t_thick


def _plot_misc(ax_line, data: ndarray, t: ndarray, excursion: int, highlight: int):
    """
    Format the display of synthetic angle gather.

    :param ax_line: Plot axes
    :param data: Synthetic traces generated from convolved zoeppritz.
    :param t: Regularly spaced ime samples
    :param excursion: Adjust plot width
    :param highlight:
    """
    import numpy as np
    import matplotlib.pyplot as plt

    [n_trace, _] = data.shape

    t = np.hstack([0, t, t.max()])

    for i in range(0, n_trace):
        _t = excursion * data[i] / np.max(np.abs(data)) + i
        _t = np.hstack([i, _t, i])

        if i == highlight:
            lw = 2
        else:
            lw = 0.5

        ax_line.plot(_t, t, color='black', linewidth=lw)

        plt.fill_betweenx(t, _t, i, where=_t > i, facecolor=[0.6, 0.6, 1.0], linewidth=0)
        plt.fill_betweenx(t, _t, i, where=_t < i, facecolor=[1.0, 0.6, 0.6], linewidth=0)

    ax_line.set_xlim((-excursion, n_trace + excursion))
    ax_line.xaxis.tick_top()
    ax_line.xaxis.set_label_position('top')
    ax_line.invert_yaxis()


def wedge_model(syn_zo: ndarray, layer_times: ndarray, t: ndarray, t_min: float,
                t_max: float, h_max: float, h_step: float, excursion: int, dt=0.0001):
    """
    Plot a three layer wedge model amplitudes and zero offset seismogram.

    :param syn_zo: Synthetic Seismogram
    :param layer_times: travel time to reflectors
    :param t: regularly sampled time series defining model sampling
    :param t_min: minimum time duration
    :param t_max: maximum time duration
    :param h_max: maximum depth
    :param h_step: depth steps, default is 1
    :param excursion: adjust plot width
    :param dt: trace parameter, changing this from 0.0001 can affect the display quality.
    """
    [n_trace, _] = syn_zo.shape
    t_trace = _tuning_trace(syn_zo=syn_zo)
    t_thick = _tuning_thickness(syn_zo=syn_zo)
    layer_index = np.array(np.round(layer_times / dt), dtype='int16')

    fig = plt.figure(figsize=(10, 12))
    fig.set_facecolor('white')

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(layer_times[:, 0], color='blue', lw=1.5)
    ax0.plot(layer_times[:, 1], color='red', lw=1.5)
    ax0.set_ylim((t_min, t_max))
    ax0.invert_yaxis()
    ax0.set_title('Three-layer wedge model', pad=20, fontsize=15)
    ax0.set_xlabel('Thickness (m)')
    ax0.set_ylabel('Time (s)')
    plt.text(2, t_min + (layer_times[0, 0] - t_min) / 2., 'Layer A', fontsize=16)
    plt.text(h_max / h_step - 2, layer_times[-1, 0] + (layer_times[-1, 1] - layer_times[-1, 0]) / 2., 'Layer B',
             fontsize=16,
             horizontalalignment='right')
    plt.text(2, layer_times[0, 0] + (t_max - layer_times[0, 0]) / 2., 'Layer C', fontsize=16)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    ax0.set_xlim((-excursion, n_trace + excursion))

    ax1 = fig.add_subplot(gs[1])
    _plot_misc(ax_line=ax1, data=syn_zo, t=t, excursion=excursion, highlight=t_trace)
    ax1.plot(layer_times[:, 0], color='blue', lw=1.5)
    ax1.plot(layer_times[:, 1], color='red', lw=1.5)
    ax1.set_title('Normal polarity zero-offset synthetic seismogram', pad=20, fontsize=15)
    ax1.set_ylim((t_min, t_max))
    ax1.invert_yaxis()
    ax1.set_xlabel('Thickness (m)')
    ax1.set_ylabel('Time (s)')

    ax2 = fig.add_subplot(gs[2])
    ax2.grid()
    ax2.plot(syn_zo[:, layer_index[:, 0]], color='blue')
    ax2.set_xlim((-excursion, n_trace + excursion))
    ax2.axvline(t_trace, color='k', lw=2)
    ax2.set_title('Amplitude of synthetic at upper interface', pad=20, fontsize=15)
    ax2.set_xlabel('Thickness (m)')
    ax2.set_ylabel('Amplitude')
    plt.text(t_trace + 2, plt.ylim()[0] * 1.1,
             'tuning thickness = {0} m'.format(str(t_thick)),
             fontsize=16)
    plt.tight_layout()
    plt.show()
