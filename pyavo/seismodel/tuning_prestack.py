"""
Functions to generate a synthetic angle gather from a 3-layer property model to examine pre-stack tuning effects.
"""

import math
from pyavo.seismodel.tuning_wedge import *


def int_depth(h_int: list, thickness: float):
    """
    Computes depth to the interface for a three layer model.

    :param h_int: depth to to first interface
    :param thickness: thickness
    :return:
        d_interface: interface depths
    """
    d_interface = h_int
    d_interface.append(d_interface[0] + thickness)
    return d_interface


def ray_param(v_int: float, theta: float) -> float:
    """
    Calculates the ray parameter P.

    :param v_int: Interval velocity
    :param theta: Angle of incidence (deg)
    :return:
        p: Ray parameter
    """

    # Cast inputs to floats
    theta = float(theta)
    v = float(v_int)

    p = math.sin(math.radians(theta)) / v  # ray parameter calculation

    return p


def rc_zoep(vp1: float, vs1: float, vp2: float, vs2: float,
            rho1: float, rho2: float, theta1: float) -> ndarray:
    """
    Calculate the Reflection & Transmission coefficients using Zoeppritz equations.

    :param vp1: P-wave velocity in upper layer
    :param vs1: S-wave velocity in upper layer
    :param vp2: P-wave velocity in lower layer
    :param vs2: S-wave velocity in lower layer
    :param rho1: Density of upper layer
    :param rho2: Density of lower layer
    :param theta1: Angle of incidence of ray
    :return:
        R: Reflection coefficients

    Reference:
        The Rock Physics Handbook, Dvorkin et al.
    """
    # Cast inputs to floats
    vp1 = float(vp1)
    vp2 = float(vp2)
    vs1 = float(vs1)
    vs2 = float(vs2)
    rho1 = float(rho1)
    rho2 = float(rho2)
    theta1 = float(theta1)

    # Calculate reflection & transmission angles
    theta1 = math.radians(theta1)  # Convert theta1 to radians
    p = ray_param(vp1, math.degrees(theta1))  # Ray parameter
    theta2 = math.asin(p * vp2)  # Transmission angle of P-wave
    phi1 = math.asin(p * vs1)  # Reflection angle of converted S-wave
    phi2 = math.asin(p * vs2)  # Transmission angle of converted S-wave

    # Matrix form of Zoeppritz Equations... M & N are two of the matricies
    M = np.array([
        [-math.sin(theta1), -math.cos(phi1), math.sin(theta2), math.cos(phi2)],
        [math.cos(theta1), -math.sin(phi1), math.cos(theta2), -math.sin(phi2)],
        [2 * rho1 * vs1 * math.sin(phi1) * math.cos(theta1), rho1 * vs1 * (1 - 2 * math.sin(phi1) ** 2),
         2 * rho2 * vs2 * math.sin(phi2) * math.cos(theta2), rho2 * vs2 * (1 - 2 * math.sin(phi2) ** 2)],
        [-rho1 * vp1 * (1 - 2 * math.sin(phi1) ** 2), rho1 * vs1 * math.sin(2 * phi1),
         rho2 * vp2 * (1 - 2 * math.sin(phi2) ** 2), -rho2 * vs2 * math.sin(2 * phi2)]
    ], dtype='float')

    N = np.array([
        [math.sin(theta1), math.cos(phi1), -math.sin(theta2), -math.cos(phi2)],
        [math.cos(theta1), -math.sin(phi1), math.cos(theta2), -math.sin(phi2)],
        [2 * rho1 * vs1 * math.sin(phi1) * math.cos(theta1), rho1 * vs1 * (1 - 2 * math.sin(phi1) ** 2),
         2 * rho2 * vs2 * math.sin(phi2) * math.cos(theta2), rho2 * vs2 * (1 - 2 * math.sin(phi2) ** 2)],
        [rho1 * vp1 * (1 - 2 * math.sin(phi1) ** 2), -rho1 * vs1 * math.sin(2 * phi1),
         -rho2 * vp2 * (1 - 2 * math.sin(phi2) ** 2), rho2 * vs2 * math.sin(2 * phi2)]
    ], dtype='float')

    # This is the important step, calculating coefficients for all modes and rays
    R = np.dot(np.linalg.inv(M), N)

    return R


def n_angles(theta1_min: float, theta1_max: float, theta1_step=1) -> int:
    """
    Computes number of traces for given angles on incidence.

    :param theta1_min: minimum incidence angle
    :param theta1_max: maximum incidence angle
    :param theta1_step: angle of incidence steps, default is 1
    :return:
        n_trace: number of traces
    """
    n_trace = int((theta1_max - theta1_min) / theta1_step + 1)
    return n_trace


def calc_theta_rc(theta1_min: float, theta1_step: float, vp: list,
                  vs: list, rho: list, ang) -> tuple:
    """
    Computes the reflection coefficients for given incidence angles in a three layer model.

    :param theta1_min: minimum incidence angle
    :param theta1_step: maximum incidence angle
    :param vp: Layer P-wave velocities
    :param vs: Layer S-wave velocities
    :param rho: Density of layers
    :param ang: Angle of incidence
    :return:
        theta1_samp : n-samples of incidence angles,
        rc_1: Reflection coefficient at layer1 / layer2,
        rc_2: Reflection coefficient at layer2 / layer3.

    """
    theta1_samp = theta1_min + theta1_step * ang
    rc_1 = rc_zoep(vp[0], vs[0], vp[1], vs[1], rho[0], rho[1], theta1_samp)
    rc_2 = rc_zoep(vp[1], vs[1], vp[2], vs[2], rho[1], rho[2], theta1_samp)
    return theta1_samp, rc_1, rc_2


def layer_index(lyr_times: list, dt=0.0001) -> tuple:
    """
    Calculate array indices corresponding to top/base interfaces.

    :param lyr_times: Interface times
    :param dt: change in time, default is 0.0001. Changing this from 0.00005 can affect the display quality.

    :return:
        lyr1_indx: array
            Layer 1
        lyr2_indx: array
            Layer 2
    """
    lyr_t = np.array(lyr_times)
    lyr_indx = np.array(np.round(lyr_t / dt), dtype='int16')
    lyr1_indx = list(lyr_indx[:, 0])
    lyr2_indx = list(lyr_indx[:, 1])
    return lyr1_indx, lyr2_indx


def avo_inv(rc_zoep, ntrc: int, top: list) -> dict:
    """
    AVO inversion for INTERCEPT and GRADIENT from analytic and convolved reflectivity values.
    Linear least squares method is used for estimating INTERCEPT and GRADIENT coefficients.

    :param rc_zoep: Zoeppritz reflection coefficient values
    :param ntrc: Number of traces
    :param top: Convolved top reflectivity values
    :return: Zoeppritz and Convolved reflectivities
    """
    Yzoep = np.array(rc_zoep[:, 0])
    Yzoep = Yzoep.reshape((ntrc, 1))

    Yconv = np.array(top)
    Yconv = Yconv.reshape((ntrc, 1))

    ones = np.ones(ntrc)
    ones = ones.reshape((ntrc, 1))

    sintheta2 = np.sin(np.radians(np.arange(0, ntrc))) ** 2
    sintheta2 = sintheta2.reshape((ntrc, 1))

    X = np.hstack((ones, sintheta2))

    #   ... matrix solution of normal equations
    Azoep = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Yzoep)
    Aconv = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Yconv)

    return {'Zoeppritz': (Azoep[0], Azoep[1]),
            'Convolved': (Aconv[0], Aconv[1])
            }


def t_domain(t, vp: list, vs: list, rho: list, lyr1_index: list, lyr2_index: list) -> tuple:
    """
    Create a "digital" time domain version of an input property model.

    :param t: array of regularly spaced time samples
    :param vp: P-wave velocity
    :param vs: S-wave velocity
    :param rho: Density of layers
    :param lyr1_index: Array indices of top interface
    :param lyr2_index: Array indices of base interface
    :return:
        vp_dig: P-wave velocity in digital t-domain,
        vs_dig: S-wave velocity in digital t-domain,
        rho_dig: Density in digital t-domain.
    """
    vp_dig = np.zeros(t.shape)
    vs_dig = np.zeros(t.shape)
    rho_dig = np.zeros(t.shape)
    P = lyr1_index[0]
    Q = lyr2_index[0]

    vp_dig[0:P] = vp[0]
    vp_dig[P:Q] = vp[1]
    vp_dig[Q:] = vp[2]

    vs_dig[0:P] = vs[0]
    vs_dig[P:Q] = vs[1]
    vs_dig[Q:] = vs[2]

    rho_dig[0:P] = rho[0]
    rho_dig[P:Q] = rho[1]
    rho_dig[Q:] = rho[2]

    return vp_dig, vs_dig, rho_dig


def plot_vawig(axhdl, data, t, excursion):
    """
    Format the display of synthetic angle gather.

    :param axhdl: Plot axes
    :param data: Synthetic traces generated from convolved zoeppritz.
    :param t: Regularly spaced ime samples
    :param excursion: Adjust plot width
    """

    [ntrc, nsamp] = data.shape

    t = np.hstack([0, t, t.max()])

    for i in range(0, ntrc):
        tbuf = excursion * data[i, :] / np.max(np.abs(data)) + i

        tbuf = np.hstack([i, tbuf, i])

        axhdl.plot(tbuf, t, color='black', linewidth=0.5)
        plt.fill_betweenx(t, tbuf, i, where=tbuf > i, facecolor=[0.6, 0.6, 1.0], linewidth=0)
        plt.fill_betweenx(t, tbuf, i, where=tbuf < i, facecolor=[1.0, 0.7, 0.7], linewidth=0)

    axhdl.set_xlim((-excursion, ntrc + excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()


#   Create the plot figure
def syn_angle_gather(min_time: float, max_time: float, lyr_times: ndarray, thickness: float,
                     top_ref: list, base_ref: list, vp_dig: ndarray, vs_dig: ndarray, rho_dig: ndarray,
                     syn_zoep: ndarray, rc_zoep: ndarray, t: ndarray, excursion: int):
    """
    Plot synthetic angle gather for three layer model displayed in normal polarity and amplitudes extracted
    along the upper and lower interfaces.

    :param min_time: Minimum plot time (s)
    :param max_time: Maximum plot time (s)
    :param lyr_times: interface times
    :param thickness: apparent thickness of second layer
    :param top_ref: convolved top reflectivity values
    :param base_ref: convolved base reflectivity values
    :param vp_dig: P-wave velocity in digital time domain
    :param vs_dig: S-wave velocity in digital time domain
    :param rho_dig: Density of layers in digital time domain
    :param syn_zoep: Synthetic seismogram
    :param rc_zoep: Zoeppritz reflectivities
    :param t: regularly spaced time samples
    :param excursion: Adjust plot width
    """
    fig = plt.figure(figsize=(16, 12))
    fig.set_facecolor('white')
    [ntrc, nsamp] = syn_zoep.shape

    #   Plot log curves in two-way time
    ax0a = fig.add_subplot(261)
    l_vp_dig, = ax0a.plot(vp_dig / 1000, t, 'k', lw=2)
    ax0a.set_ylim((min_time, max_time))
    ax0a.set_xlim(1.5, 4.0)
    ax0a.invert_yaxis()
    ax0a.set_ylabel('TWT (sec)')
    ax0a.xaxis.tick_top()
    ax0a.xaxis.set_label_position('top')
    ax0a.set_xlabel('Vp (km/s)')
    ax0a.axhline(lyr_times[0, 0], color='blue', lw=2, alpha=0.5)
    ax0a.axhline(lyr_times[0, 1], color='red', lw=2, alpha=0.5)
    ax0a.grid()

    ax0b = fig.add_subplot(262)
    l_vs_dig, = ax0b.plot(vs_dig / 1000, t, 'k', lw=2)
    ax0b.set_ylim((min_time, max_time))
    ax0b.set_xlim((0.8, 2.0))
    ax0b.invert_yaxis()
    ax0b.xaxis.tick_top()
    ax0b.xaxis.set_label_position('top')
    ax0b.set_xlabel('Vs (km/s)')
    ax0b.set_yticklabels('')
    ax0b.axhline(lyr_times[0, 0], color='blue', lw=2, alpha=0.5)
    ax0b.axhline(lyr_times[0, 1], color='red', lw=2, alpha=0.5)
    ax0b.grid()

    ax0c = fig.add_subplot(263)
    l_rho_dig, = ax0c.plot(rho_dig, t, 'k', lw=2)
    ax0c.set_ylim((min_time, max_time))
    ax0c.set_xlim((1.6, 2.6))
    ax0c.invert_yaxis()
    ax0c.xaxis.tick_top()
    ax0c.xaxis.set_label_position('top')
    ax0c.set_xlabel('Density')
    ax0c.set_yticklabels('')
    ax0c.axhline(lyr_times[0, 0], color='blue', lw=2, alpha=0.5)
    ax0c.axhline(lyr_times[0, 1], color='red', lw=2, alpha=0.5)
    ax0c.grid()

    plt.text(2.55,
             min_time + (lyr_times[0, 0] - min_time) / 2.,
             'Layer 1',
             fontsize=13,
             horizontalalignment='right')
    plt.text(2.55,
             lyr_times[0, 1] + (lyr_times[0, 0] - lyr_times[0, 1]) / 2. + 0.002,
             'Layer 2',
             fontsize=13,
             horizontalalignment='right')
    plt.text(2.55,
             lyr_times[0, 0] + (max_time - lyr_times[0, 0]) / 2.,
             'Layer 3',
             fontsize=13,
             horizontalalignment='right')

    #   Plot synthetic gather and model top & base interfaces in two-way time
    ax1 = fig.add_subplot(222)
    plot_vawig(ax1, syn_zoep, t, excursion)
    ax1.set_ylim((min_time, max_time))
    l_int1, = ax1.plot(lyr_times[:, 0], color='blue', lw=2)
    l_int2, = ax1.plot(lyr_times[:, 1], color='red', lw=2)

    plt.legend([l_int1, l_int2], ['Interface 1', 'Interface 2'], loc=4)
    ax1.invert_yaxis()
    label_str = 'Synthetic angle gather\nLayer 2 thickness = %4.1fm' % thickness
    ax1.set_xlabel(label_str, fontsize=13)
    ax1.set_ylabel('TWT (sec)')

    #   Plot Zoeppritz and convolved reflectivity curves
    ax2 = fig.add_subplot(2, 2, 3)

    l_syn1, = ax2.plot(top_ref, color='blue', linewidth=2)
    l_rc1, = ax2.plot(rc_zoep[:, 0], '--', color='blue', lw=2)

    ax2.set_xlim((-excursion, ntrc + excursion))
    ax2.grid()
    ax2.set_xlabel('Angle of incidence (deg)')
    ax2.set_ylabel('Reflection coefficient')
    ax2.set_title('Upper interface reflectivity')
    plt.legend([l_syn1, l_rc1], ['Convolved', 'Zoepprtiz'], loc=0)

    ax3 = fig.add_subplot(2, 2, 4)
    l_syn2, = ax3.plot(base_ref, color='red', linewidth=2)
    l_rc2, = ax3.plot(rc_zoep[:, 1], '--', color='red', lw=2)
    ax3.set_xlim((-excursion, ntrc + excursion))
    ax3.grid()
    ax3.set_xlabel('Angle of incidence (deg)')
    ax3.set_ylabel('Reflection coefficient')
    ax3.set_title('Lower interface reflectivity')
    plt.legend([l_syn2, l_rc2], ['Convolved', 'Zoepprtiz'], loc=0)
    #   Display the plot
    plt.show()
