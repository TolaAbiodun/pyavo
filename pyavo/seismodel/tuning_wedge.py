# -*- coding: utf-8 -*-
"""
Functions to generate a three layer wedge model using the reflection coefficients and travel times
Original Script by Wes Hamlyn, 2014
Refactored by Tola Abiodun, 2021.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def get_rc(Vp, rho):
    """
    Calculates the Reflection Coefficient at the interface separating two layers.

    The ratio of amplitude of the reflected wave to the incident wave, or how much energy is reflected.

    If the wave has normal incidence, then its reflection coefficient can be expressed as:

    R = (ρ2V2 − ρ1V1) / (ρ2V2 + ρ1V1), where:

    R = reflection coefficient, whose values range from −1 to +1
    ρ1 = density of medium 1
    ρ2 = density of medium 2
    V1 = velocity of medium 1
    V2 = velocity of medium 2.

    Typical values of R are approximately −1 from water to air, meaning that nearly 100% of the energy is reflected and none is transmitted; ~0.5 from water to rock; and ~0.2 for shale to sand.

    At non-normal incidence, the reflection coefficient defined as a ratio of amplitudes depends on other parameters, such as the shear velocities, and is described as a function of incident angle by the Zoeppritz equations.

    :param Vp: P-wave velocity (m/s)
    :param rho: layer density(g/cc)
    :return: List
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
    :return: List

    Usage
    ---------------
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
    Create regularly sampled time series defining model sampling
    :param t_min: Minimum time duration
    :param t_max: Maximum time duration
    :param dt: Change in time, default = 0.0001
    :return: List
    """
    n_samp = int((t_max - t_min) / dt) + 1
    time = []
    for t in range(0, n_samp):
        time.append(t * dt)
    return time


def mod_digitize(rc_int: list, t_int: list, t: list) -> list:
    """
    Digitize a 3 layer model using reflection coefficients and interface times

    Usage
    ----------------------
    rc = digitize_model(rc_int, t_int, t)

    :param rc_int: reflection coefficients corresponding to interface times
    :param t_int: interface times
    :param t: regularly sampled time series defining model sampling
    :return: list
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

def syn_seis(ref_coef:list, wav_amp):
    smg = np.convolve(ref_coef, wav_amp, mode='same')
    smg = list(smg)
    return smg

def int_depth(h_int:list, dh_min:float, dh_step:float):
    d_interface = h_int
    d_interface.append(d_interface[0]+dh_min+dh_step*model)
    return d_interface

def n_model(h_min=float, h_max=float, h_step=1):
    n_trace = int((h_max - h_min) / h_step + 1)
    return n_trace

def ricker(sample_rate, length, c_freq):
    """
    Generate time and amplitude values for a zero-phase wavelet.

    The second derivative of the Gaussian function or the third derivative of the normal-probability density function.

    A Ricker wavelet is often used as a zero-phase embedded wavelet in modeling and synthetic seismogram manufacture. Norman H. Ricker (1896–1980), American geophysicist.

    :param sample_rate: sample rate in seconds
    :param length: length of time (dt) in seconds
    :param c_freq: central frequency of wavelet (cycles/seconds or Hz).
    :return: ndarray

    Example:
    -------
    time, wavelet = (sample_rate,duration,c_freq)
    """

    t_min = -length / 2
    t_max = (length - sample_rate) / 2
    wv_time = np.linspace(t_min, t_max, int(length / sample_rate))
    wv_amp = (1. - 2. * (np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2)) * np.exp(
        -(np.pi ** 2) * (c_freq ** 2) * (wv_time ** 2))

    return wv_time, wv_amp


# Calculate the tuning trace and tuning thickness
def tuning_trace(syn_zo):
    t_trace = np.argmax(np.abs(syn_zo.T)) % syn_zo.T.shape[1]
    return t_trace


def tuning_thickness(syn_zo, step):
    t_thick = tuning_trace(syn_zo) * step
    return t_thick

def plot_misc(ax_line, data, t, excursion, highlight):
    import numpy as np
    import matplotlib.pyplot as plt

    [n_trace, n_samp] = data.shape

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

    ax_line.set_xlim((-excursion, n_trace+excursion))
    ax_line.xaxis.tick_top()
    ax_line.xaxis.set_label_position('top')
    ax_line.invert_yaxis()


def wedge_model(syn_zo, layer_times, t, t_min, t_max, h_min, h_max, h_step, excursion, dt=0.0001):
    """
    Plot a three layer wedge model amplitudes and zero offset seismogram.

    :param syn_zo: Synthetic Seismogram
    :param layer_times: travel time to reflectors
    :param t: regularly sampled time series defining model sampling
    :param t_min:minimum time duration
    :param t_max: maximum time duration
    :param h_min:
    :param h_max:
    :param h_step:
    :param excursion:
    :param dt:
    :return:

    """
    [n_trace, n_sample] = syn_zo.shape
    t_trace = tuning_trace(syn_zo=syn_zo)
    t_thick = tuning_thickness(syn_zo=syn_zo, step=1)
    layer_index = np.array(np.round(layer_times / dt), dtype='int16')

    fig = plt.figure(figsize=(10, 12))
    fig.set_facecolor('white')

    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(layer_times[:, 0], color='blue', lw=1.5)
    ax0.plot(layer_times[:, 1], color='red', lw=1.5)
    ax0.set_ylim((t_min, t_max))
    ax0.invert_yaxis()
    ax0.set_title('Three-layer wedgde model', pad=20, fontsize=15)
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
    plot_misc(ax_line=ax1, data=syn_zo, t=t, excursion=excursion, highlight=t_trace)
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

    plt.savefig('tuning_wedge_model.png')
    plt.tight_layout()
    plt.show()


# Data
lyr_times = []
refl_z = []
syn_zo = []
dz_step = 1
dz_min = 0.0
rho_mod = [1.95, 2.0, 1.98]
vp_mod = [2500.0, 2600.0, 2550.0]

# Create model
nmodel = n_model(h_min=0.0, h_max=60.0)

#   Calculate reflectivities from model parameters
rc_int = get_rc(Vp=vp_mod, rho=rho_mod)

#   Generate ricker wavelet
wlt_time, wlt_amp = ricker(sample_rate=0.0001, length=0.128, c_freq=50.0)

for model in range(0, nmodel):
    # Calculate Interface Depths
    z_int = int_depth(h_int=[500.0], dh_min=0.0, dh_step=1)

    #   Calculate interface times
    t_int = calc_times(z_int, vp_mod)
    lyr_times.append(t_int)

    #   Digitize 3-layer model
    t_samp = time_samples(t_min=0, t_max=0.5)

    rc = mod_digitize(rc_int, t_int, t_samp)
    refl_z.append(rc)

    #   Convolve wavelet with reflectivity
    s = syn_seis(ref_coef=rc, wav_amp=wlt_amp)
    syn_zo.append(s)

# Cast the Sythetic Seismogram to a Numpy Array
syn_zo = np.array(syn_zo)
t_samp = np.array(t_samp)

#Convert layer times to NdArray
lyr_times = np.array(lyr_times)
lyr_times.shape

print(t_int)
print(z_int)
print(f'layer times is {lyr_times}')
print(rc_int)
print(syn_zo.shape)
print(type(t_samp))

wedge_model(syn_zo=syn_zo, layer_times=lyr_times, t=t_samp,t_min=0.1,t_max=0.3,h_min=0.0,h_max=60.0,h_step=1,excursion=2)
