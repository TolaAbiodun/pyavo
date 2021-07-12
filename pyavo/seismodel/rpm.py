# -*- coding: utf-8 -*-
"""
Functions to create rock physics models and apply them to
well log analysis to quantify seismic responses

Created by: Tola Abiodun, 2021

Using equations http://www.subsurfwiki.org/wiki/Elastic_modulus
from Mavko, G, T Mukerji and J Dvorkin (2003), The Rock Physics Handbook, Cambridge University Press
"""

# import libraries
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import Union


# Function to explore key input logs for Rock physics modeling
def plot_log(data: DataFrame, vsh: Union[float, iter], vp: Series, vs: Series, imp: list, vp_vs: list,
             phi: list, rho: list, z_init: Union[int, float],
             z_final: Union[int, float], shale_cutoff: Union[int, float],
             sand_cutoff: Union[int, float]):
    """
    Explore key input logs for Rock physics modeling. Generates a log plot of the shale volume, acoustic impedance,
    Vp/Vs and a cross plot of Porosity against P-wave velocity and acoustic impedance against Vp/VS.

    :param data: pandas DataFrame
    :param vsh: Shale volume as input from vshale log
    :param vp: Compressional wave velocity log
    :param vs: Shear wave velocity log
    :param imp: Acoustic impedance from acoustic impedance log
    :param vp_vs: Vp/Vs ratios
    :param phi: porosity values form PHI log
    :param rho: density values from RHOB log
    :param z_init: minimum depth from depth log
    :param z_final: maximum depth from depth log
    :param shale_cutoff: Shale cutoff
    :param sand_cutoff: Sand cutoff
    :return: log plots(Vp/Vs, AI, Vsh) and cross plots(AI,Vp/Vs & PHI,Vp)
    """
    # Default plotting parameters
    sand_ppt = {'marker': 'o',
                'color': 'gold',
                'alpha': 0.5,
                'ls': 'none',
                'ms': 5,
                'mec': 'none'}

    shale_ppt = {'marker': 'o',
                 'color': 'saddlebrown',
                 'alpha': 0.5,
                 'ls': 'none',
                 'ms': 5,
                 'mec': 'none'}

    log_ppt = {'lw': 0.9,
               'ls': '-',
               'color': 'black'}

    depth = data.index  # Grab depth values from index of dataframe
    sand = (depth >= z_init) & (depth <= z_final) & (vsh <= sand_cutoff)
    shale = (depth >= z_init) & (depth <= z_final) & (vsh >= shale_cutoff)

    # make log plots and cross plots
    f = plt.subplots(figsize=(14, 5))
    ax0 = plt.subplot2grid((1, 9), (0, 0), colspan=1)  # Vp/Vs log
    ax1 = plt.subplot2grid((1, 9), (0, 1), colspan=1)  # Vsh log
    ax2 = plt.subplot2grid((1, 9), (0, 2), colspan=1)  # AI log
    ax3 = plt.subplot2grid((1, 9), (0, 3), colspan=3)  # crossplot phi - vp
    ax4 = plt.subplot2grid((1, 9), (0, 6), colspan=3)  # crossplot ip - vp/vs

    ax0.plot(vp_vs[sand], depth[sand], **sand_ppt)
    ax0.plot(vp_vs[shale], depth[shale], **shale_ppt)
    ax0.plot(vp_vs, depth, **log_ppt)
    ax0.set_xlabel('Vp/Vs')
    ax0.set_xlim(min(vp_vs), max(vp_vs))

    ax1.plot(vsh[sand], depth[sand], **sand_ppt)
    ax1.plot(vsh[shale], depth[shale], **shale_ppt)
    ax1.plot(vsh, depth, **log_ppt)
    ax1.set_xlabel('VSH')

    ax2.plot(imp[sand], depth[sand], **sand_ppt)
    ax2.plot(imp[shale], depth[shale], **shale_ppt)
    ax2.plot(imp, depth, **log_ppt)
    ax2.set_xlabel('AI')
    ax2.set_xlim(min(imp), max(imp))

    ax3.plot(phi[sand], vp[sand], **sand_ppt)
    ax3.set_xlim(min(phi), max(phi))
    ax3.set_ylim(min(depth), max(depth) * 1.5)
    ax3.set_xlabel('x=PHI, y=Vp')

    ax4.plot(vp * rho[sand], vp / vs[sand], **sand_ppt)
    ax4.plot(vp * rho[shale], vp / vs[shale], **shale_ppt)
    ax4.set_xlim(min(imp), max(imp)), ax4.set_ylim(min(vp_vs), max(vp_vs))
    ax4.set_xlabel('x=AI, y=Vp/Vs')

    # Adjust the subplot layout parameters.
    plt.subplots_adjust(left=0.03, right=0.95, wspace=1.0)

    # DEPTH
    for axes in [ax0, ax1, ax2]:
        axes.set_ylim(z_final, z_init)

    # SET TICK LABELS
    for axes in [ax0, ax1, ax2, ax3, ax4]:
        axes.tick_params(which='major', labelsize=8)

    for axes in [ax1, ax2]:
        axes.set_yticklabels([])

    plt.show()


# Function to create rock physics models
def soft_sand(k_min, mu_min, phi: ndarray, cd_num=8.6, cp=0.4, P=10, f=1):
    """
    Computes the bulk modulus and shear modulus of soft-sand (uncemented) rock physics model.

    :param k_min: bulk modulus of mineral (Gpa)
    :param mu_min: shear modulus of mineral (Gpa)
    :param phi: porosity (fraction) expressed as vectors
    :param cd_num: coordination number
    :param cp: critical porosity (random close pack of well-sorted rounded quartz grains)
    :param P: Confining pressure/rock stress (Gpa)
    :param f: shear modulus correction factor
              0 =  dry frictionless packing
              1 = dry pack with perfect adhesion
    :return: Bulk modulus and Shear modulus

    Reference:
        The Rock Physics Handbook: Tools for Seismic Analysis of Porous Media - Gary M. Mavko, Jack Dvorkin,
        and Tapan Mukerji
    """
    k_hm, mu_hm = hz_mindlin(k_min, mu_min, phi, cd_num, cp, P, f)
    k_dry = -4 / 3 * mu_hm + (((phi / cp) / (k_hm + 4 / 3 * mu_hm)) + ((1 - phi / cp) / (k_min + 4 / 3 * mu_hm))) ** -1
    t = mu_hm / 6 * ((9 * k_hm + 8 * mu_hm) / (k_hm + 2 * mu_hm))
    mu_dry = -t + ((phi / cp) / (mu_hm + t) + ((1 - phi / cp) / (mu_min + t))) ** -1
    return k_dry, mu_dry


def stiff_sand(k_min, mu_min, phi: ndarray, cd_num=8.6, cp=0.4, P=10, f=1):
    """
    Computes the bulk modulus and shear modulus of stiff-sand (uncemented) rock physics model

    :param k_min: bulk modulus of mineral (Gpa)
    :param mu_min: shear modulus of mineral (Gpa)
    :param phi: porosity (fraction) expressed as vectors
    :param cd_num: coordination number
    :param cp: critical porosity (random close pack of well-sorted rounded quartz grains)
    :param P: Confining pressure/rock stress (Gpa)
    :param f: shear modulus correction factor
              0 =  dry frictionless packing
              1 = dry pack with perfect adhesion
    :return: Bulk modulus and Shear modulus

    Reference:
        The Rock Physics Handbook: Tools for Seismic Analysis of Porous Media - Gary M. Mavko,
        Jack Dvorkin, and Tapan Mukerji
    """
    k_hm, mu_hm = hz_mindlin(k_min, mu_min, phi, cd_num, cp, P, f)
    k_dry = -4 / 3 * mu_min + (
            ((phi / cp) / (k_hm + 4 / 3 * mu_min)) + ((1 - phi / cp) / (k_min + 4 / 3 * mu_min))) ** -1
    t = mu_min / 6 * ((9 * k_min + 8 * mu_min) / (k_min + 2 * mu_min))
    mu_dry = -t + ((phi / cp) / (mu_hm + t) + ((1 - phi / cp) / (mu_min + t))) ** -1
    return k_dry, mu_dry


def hz_mindlin(k_min, mu_min, phi: ndarray, cd_num=8.6, cp=0.4, P=10, f=1):
    """
    Computes the bulk modulus and shear modulus of the Hertz-Mindlin rock physics model

    :param k_min: bulk modulus of mineral (Gpa)
    :param mu_min: shear modulus of mineral (Gpa)
    :param phi: porosity (fraction) expressed as vector values
    :param cd_num: coordination number
    :param cp: critical porosity (random close pack of well-sorted rounded quartz grains)
    :param P: Confining pressure/rock stress (Mpa)
    :param f: shear modulus correction factor
        0 =  dry frictionless packing
        1 = dry pack with perfect adhesion
    :return: Bulk modulus and Shear modulus

    Reference
      The Rock Physics Handbook: Tools for Seismic Analysis of Porous Media - Gary M. Mavko,
      Jack Dvorkin, and Tapan Mukerji
      """
    # Convert pressure from MPa to Gpa
    P = P / 1e3
    # compute the poisson's ratio of mineral mixture
    pr = (3 * k_min - 2 * mu_min) / (6 * k_min + 2 * mu_min)

    # compute the bulk modulus
    k_hzm = (P * (cd_num ** 2 * (1 - cp) ** 2 * mu_min ** 2) / (18 * np.pi ** 2 * (1 - pr) ** 2)) ** (1 / 3)

    # compute the shear modulus
    mu_hzm = ((2 + 3 * f - pr * (1 + 3 * f)) / (5 * (2 - pr))) * (
        (P * (3 * cd_num ** 2 * (1 - cp) ** 2 * mu_min ** 2) / (2 * np.pi ** 2 * (1 - pr) ** 2))) ** (1 / 3)
    return k_hzm, mu_hzm


def voigt_reuss(mod1: float, mod2: float, vfrac: float) -> tuple:
    """
    Computes the Voigt-Reuss-Hill bounds and average for a 2-component mixture.
    The Voigt Bound is obtained by assuming the strain field remains constant
    throughout the material when subjected to an arbitrary average stress field.
    The Reuss Bound is obtained by assuming the stress field remains constant
    throughout the material in an arbitrary average strain field.

    :param mod1: elastic modulus of first mineral
    :param mod2: elastic modlus of second mineral
    :param vfrac: volumetric fraction of first mineral (net-to-gross)
    :return: upper bound, lower bound, Voigt-Ruess-Hill average

    Reference
        Mavko, G, T. Mukerji and J. Dvorkin (2009), The Rock Physics Handbook: Cambridge University Press.
    """
    voigt = vfrac * mod1 + (1 - vfrac) * mod2
    reuss = 1 / (vfrac / mod1 + (1 - vfrac) / mod2)
    vrh = (voigt + reuss) / 2
    return voigt, reuss, vrh


def vel_sat(k_min: float, rho_min: float, k_fl: float, rho_fl: float,
            k_frame: float, mu_frame: float, phi: ndarray) -> tuple:
    """
    Computes velocities and densities of saturated rock using Gassmann's fluid substitution equations.

    :param k_min: bulk modulus of mineral (Gpa)
    :param rho_min: density of mineral (g/cc)
    :param k_fl: bulk modulus of fluid (Gpa)
    :param rho_fl: density of fluid (g/cc)
    :param k_frame: bulk modulus of rock frame/dry rock (Gpa)
    :param mu_frame: shear modulus of rock frame/dry rock (Gpa)
    :param phi: porosity expressed as vector values
    :return:
        k_sat: Bulk modulus
        rho_sat: Density
        vp_sat: P-wave velocity
        vs_sat: S-wave velocity
    """
    k_sat = k_frame + (1 - k_frame / k_min) ** 2 / ((phi / k_fl) + ((1 - phi) / k_min) - (k_frame / k_min ** 2))
    rho_sat = rho_min * (1 - phi) + rho_fl * phi
    vp_sat = np.sqrt((k_sat + 4 / 3 * mu_frame) / rho_sat) * 1e3
    vs_sat = np.sqrt(k_frame / rho_sat) * 1e3
    return k_sat, rho_sat, vp_sat, vs_sat


def plot_rpm(df: DataFrame, k_qtz: float, mu_qtz: float, k_sh: float, mu_sh: float, rho_sh: float, rho_qtz: float,
             k_br: float, rho_br: float, phi: ndarray, NG: ndarray, vsh: Series, z_init: float, z_final: float,
             sand_cut: float, shale_cut: float, P: Union[int, float], cp_ssm: float, cn_ssm: float, cp_stm: float,
             cn_stm: float, eff_phi: Series, vp: Series):
    """
    Plot soft sand and stiff sand rock physics models for different mineralogies.

    :param df: Pandas DataFrame
    :param k_qtz: Bulk modulus of quartz (GPa)
    :param mu_qtz: Shear modulus of quartz (GPa)
    :param k_sh: Bulk modulus of shale (Gpa)
    :param mu_sh: Shear modulus of shale (Gpa)
    :param rho_sh: Density of shale (g/cc)
    :param rho_qtz: Density of quartz (g/cc)
    :param k_br: Bulk modulus of brine (GPa)
    :param rho_br: Density of brine (g/cc)
    :param phi: Porosity (fraction)
    :param NG: Net-to-Gross
    :param vsh: Volume of shale from VSH log
    :param z_init: minimum depth withing range of input depth log (ft)
    :param z_final:  maximum depth withing range of input depth log (ft)
    :param sand_cut: sand cutoff
    :param shale_cut: shale cutoff
    :param P: Confining pressure (Mpa)
    :param cp_ssm: Critical porosity for soft sand model
    :param cn_ssm: Coordination number for soft sand model
    :param cp_stm: Critical porosity for stiff sand model
    :param cn_stm: Coordination number for stiff sand model
    :param eff_phi: effective porosity values from porosity log (fraction)
    :param vp: P-wave velocity from VP log (m/s)
    :return: Plot of soft sand and stiff sand models with curves showing varying net-to-gross.
    """
    depth = df.index  # Grab depth values from index of dataframe

    ss = (depth >= z_init) & (depth <= z_final) & (vsh <= sand_cut)
    sh = (depth >= z_init) & (depth <= z_final) & (vsh >= shale_cut)

    sand_ppt = {'marker': 'o', 'color': 'goldenrod', 'ls': 'none', 'ms': 6, 'mec': 'none', 'alpha': 0.5}

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # For loop to plot Net-to-gross
    for i in NG:
        _, _, k_min = voigt_reuss(k_qtz, k_sh, i)
        _, _, mu_min = voigt_reuss(mu_qtz, mu_sh, i)
        rho_min = i * rho_qtz + (1 - i) * rho_sh

        # Soft Sand Model (ssm)
        kdry, mudry = soft_sand(k_min, mu_min, phi, cp=cp_ssm, cd_num=cn_ssm, P=P)
        _, _, vp_ssm, vs_ssm = vel_sat(k_min, rho_min, k_br, rho_br, kdry, mudry, phi)

        # Stiff Sand Model (stm)
        k_dry, mu_dry = stiff_sand(k_min, mu_min, phi, cp=cp_stm, cd_num=cn_stm, P=P)
        _, _, vp_stm, vs_stm = vel_sat(k_min, rho_min, k_br, rho_br, k_dry, mu_dry, phi)

        # plot subplots
        ax[0].plot(phi, vp_ssm, '-k', label='N:G={:.2f}'.format(i), alpha=i - .5)
        ax[1].plot(phi, vp_stm, '-k', label='N:G={:.2f}'.format(i), alpha=i - .5)
    for aa in ax:
        aa.plot(eff_phi[ss], vp[ss], **sand_ppt)
        aa.set_xlim(0, 0.4), aa.set_ylim(min(depth),
                                         max(depth) * 1.5)  # set maximum depth to 1.5times to show both plots
        aa.set_xlabel('Porosity')
        aa.set_ylabel('Vp', rotation=0, labelpad=15)
        aa.legend()
    ax[0].set_title('Soft Sand Model')
    ax[1].set_title('Stiff Sand Model')
    plt.show()


def plot_rpt(model='soft', fluid='oil', display=True, vsh=0.0,
             k_sh=15, k_qtz=37, mu_sh=5, mu_qtz=44, rho_sh=2.8, rho_qtz=2.6,
             k_gas=0.06, k_oil=0.9, k_br=2.8, rho_gas=0.2, rho_oil=0.8,
             rho_br=1.1, cp=0.4, cd_num=8.6, P=10, f=1):
    """
    Plot RPTs showing variations in porosity, mineralogy and fluid content defined by water saturation.

    :param model: type of rock physics model, 'soft' or 'stiff'. default = 'soft'
    :param fluid: desired fluid after Gassmann's fluid substitution, default = 'oil'
    :param display: show plot with calculated parameters, default = True
    :param vsh: volume of shale, default = 0
    :param k_sh: bulk modulus of shale, default = 15GPa
    :param k_qtz: bulk modulus of quartz, default = 37GPa
    :param mu_sh: shear modulus of shale, default = 5GPa
    :param mu_qtz: shear modulus of quartz, default = 44GPa
    :param rho_sh: density of shale, default = 2.8g/cc
    :param rho_qtz:  density of quartz, default = 2.6g/cc
    :param k_gas: bulk modulus of gas (GPa), default = 0.06GPa
    :param k_oil: bulk modulus of oil (GPa), default = 0.9GPa
    :param k_br: bulk modulus of brine (Gpa), default = 2.8GPa
    :param rho_gas: density of gas (g/cc), default = 0.2g/cc
    :param rho_oil: density of oil (g/cc), default = 0.8g/cc
    :param rho_br: density of brine (g/cc) default = 1.1g/cc
    :param cp: critical porosity, default = 0.4
    :param cd_num: coordination number, default = 8.6
    :param P: confining pressure(MPa), default = 10MPa
    :param f: shear modulus correction factor, default=1
    :return: plot of RPT with labels, Array of Acoustic impedance and Vp/Vs

    Reference:
        Quantitative Seismic Interpretation: Applying Rock Physics Tools to Reduce Interpretation Risk
        (Per Avseth, T. Mukerji and G.mavko, 2015)

    """
    # Generate poro and sw values
    phi = np.linspace(0.01, cp, 10)
    sw = np.linspace(0, 1, 10)

    # init empty array with shape using phi and sw as size
    arr1 = np.empty((phi.size, sw.size))
    arr2 = np.empty((phi.size, sw.size))

    # get initial fluid elastic properties
    (k_hc, rho_hc) = (k_oil, rho_oil) if fluid == 'oil' else (k_gas, rho_gas)

    # compute Voigt-Ruess-Hill
    _, _, k_min = voigt_reuss(k_qtz, k_sh, vsh)
    _, _, mu_min = voigt_reuss(mu_qtz, mu_sh, vsh)
    rho_min = vsh * rho_sh + (1 - vsh) * rho_qtz
    if model == 'soft':
        k_frame, mu_frame = soft_sand(k_min, mu_min, phi, cd_num, cp, P, f)
    elif model == 'stiff':
        k_frame, mu_frame = stiff_sand(k_min, mu_min, phi, cd_num, cp, P, f)

    for items, val in enumerate(sw):
        # get new fluid properties( k_fl and rho_fl)
        _, k_fl, _ = voigt_reuss(k_br, k_hc, val)
        rho_fl = val * rho_br + (1 - val) * rho_hc
        _, rho, vp, vs = vel_sat(k_min, rho_min, k_fl, rho_fl, k_frame, mu_frame, phi)
        arr1[:, items] = vp * rho  # Acoustic impedance
        arr2[:, items] = vp / vs  # Vp/Vs
    sty1 = {'backgroundcolor': 'yellow'}
    sty2 = {'backgroundcolor': 'whitesmoke',
            'ha': 'right', }

    if display:
        plt.figure(figsize=(8, 8))
        plt.plot(arr1, arr2, '-ok', alpha=0.5)
        plt.plot(arr1.T, arr2.T, '-k', alpha=0.5)
        for item, val in enumerate(phi):
            plt.text(arr1[item, -1], arr2[item, -1] + .01, '$phi={:.02f}$'.format(val), **sty1)
        plt.text(arr1[-1, 0] - 100, arr2[-1, 0], '$sw={:.02f}$'.format(sw[0]), **sty2)
        plt.text(arr1[-1, -1] - 100, arr2[-1, -1], '$sw={:.02f}$'.format(sw[-1]), **sty2)
        plt.xlabel('Acoustic Impedance', labelpad=15)
        plt.ylabel('Vp/Vs', rotation=0, labelpad=20)
        plt.title(f'Rock Physics Template (RPT) - {model.upper()}, fluid={fluid}, N:G={1 - vsh}', pad=15)

    plt.show()
    return arr1, arr2


def well_rpt(df, z_init: float, z_final: float, sand_cut: float, vsh: Series, vp: Series, vs: Series, rho: Series,
             ip_rpt0: ndarray, vpvs_rpt0: ndarray, ip_rpt1: ndarray, vpvs_rpt1: ndarray):
    """
    Plot Soft and Stiff Sand RPT models and superimpose on well data.

    :param df: Pandas Dataframe
    :param z_init: minimum depth values withing depth log range
    :param z_final: maximum depth values within depth log range
    :param sand_cut: cut off sand
    :param vsh: volume of shale from VSH log
    :param vp: P-wave velocity values from VP log
    :param vs: S-wave velocity values from VS log
    :param rho: density values from RHOB log
    :param ip_rpt0: Acoustic impedance values of first RPT model
    :param vpvs_rpt0: Vp/Vs values of first RPT model
    :param ip_rpt1: Acoustic impedance values of second RPT model
    :param vpvs_rpt1: Vp/Vs values of second RPT model
    """
    # Variables
    sand_ppt = {'marker': 'o', 'color': 'goldenrod', 'ls': 'none', 'ms': 6, 'mec': 'none', 'alpha': 0.5}
    depth = df.index  # Grab depth values from index of dataframe
    ss = (depth >= z_init) & (depth <= z_final) & (vsh <= sand_cut)

    f, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].plot(ip_rpt0, vpvs_rpt0, 'sk', mew=0, alpha=0.5)  # plot soft ss model
    ax[0].set_title('Soft Sand Model', pad=10, fontsize=16)
    ax[1].plot(ip_rpt1, vpvs_rpt1, 'sk', mew=0, alpha=0.5)  # plot stiff sand model
    ax[1].set_title('Stiff Sand Model', pad=10, fontsize=16)

    for i in ax:
        AI = vp[ss] * rho[ss]
        VPVS = vp[ss] / vs[ss]
        i.plot(AI, VPVS, **sand_ppt)
        i.set_xlim(min(AI), max(AI) * 2)
        i.set_ylim(1.0, 3.0)
        i.set_xlabel('Acoustic Impedance', labelpad=15)
        i.set_ylabel('VP/VS', rotation=0, labelpad=20)

    plt.show()


def cp_model(k_min, mu_min, phi, cp=0.4):
    """
    Build rock physics models using critical porosity concept.

    :param k_min: Bulk modulus of mineral (Gpa)
    :param mu_min: Shear modulus of mineral (Gpa)
    :param phi: Porosity (fraction)
    :param cp: critical porosity, default = 40%
    :return: Bulk and Shear modulus of rock frame

    Reference
        Critical porosity, Nur et al. (1991, 1995)
        Mavko, G, T. Mukerji and J. Dvorkin (2009), The Rock Physics Handbook: Cambridge University Press. p.353
    """
    k_frame = k_min * (1 - phi / cp)
    mu_frame = mu_min * (1 - phi / cp)
    return k_frame, mu_frame


def twolayer(n_samples: int, vp1: Union[int, float], vs1: Union[int, float],
             rho1: float, vp2: Union[int, float], vs2: Union[int, float],
             rho2: float):
    """
    Display seismic signatures at Near and Far offset traces and AVO curve for a two layer model

    :param n_samples: Number of samples
    :param vp1: P-wave velocity in top layer
    :param vs1: S-Wave velocity in top layer
    :param rho1: Density of top layer
    :param vp2: P-wave velocity in bottom layer
    :param vs2: S-wave velocity in bottom layer
    :param rho2: Density of second layer

    Reference
        Alessandro aadm (2015)
    """
    from bruges.reflection import shuey2
    from bruges.filters import ricker

    interface = int(n_samples / 2)
    ang = np.arange(31)
    wavelet = ricker(.25, 0.001, 25)

    model_ip, model_vpvs, rc0, rc1 = (np.zeros(n_samples) for i in range(4))
    model_z = np.arange(n_samples)
    model_ip[:interface] = vp1 * rho1
    model_ip[interface:] = vp2 * rho2
    model_vpvs[:interface] = np.true_divide(vp1, vs1)
    model_vpvs[interface:] = np.true_divide(vp2, vs2)

    avo = shuey2(vp1, vs1, rho1, vp2, vs2, rho2, ang)
    rc0[interface] = avo[0]
    rc1[interface] = avo[-1]
    synt0 = np.convolve(rc0, wavelet, mode='same')
    synt1 = np.convolve(rc1, wavelet, mode='same')
    clip = np.max(np.abs([synt0, synt1]))
    clip += clip * .2

    opz1 = {'color': 'k', 'linewidth': 2}
    opz2 = {'linewidth': 0, 'alpha': 0.5}

    f = plt.subplots(figsize=(14, 7))
    ax0 = plt.subplot2grid((1, 7), (0, 0), colspan=1)  # ip
    ax1 = plt.subplot2grid((1, 7), (0, 1), colspan=1)  # vp/vs
    ax2 = plt.subplot2grid((1, 7), (0, 2), colspan=1)  # synthetic @ 0 deg
    ax3 = plt.subplot2grid((1, 7), (0, 3), colspan=1)  # synthetic @ 30 deg
    ax4 = plt.subplot2grid((1, 7), (0, 4), colspan=3)  # avo curve

    ax0.plot(model_ip, model_z, **opz1)
    ax0.set_xlabel('AI')

    ax1.plot(model_vpvs, model_z, **opz1)
    ax1.set_xlabel('Vp/Vs')
    ax2.plot(synt0, model_z, **opz1)
    ax2.fill_betweenx(model_z, 0, synt0, where=synt0 > 0, facecolor='blue', **opz2)
    ax2.fill_betweenx(model_z, 0, synt0, where=synt0 < 0, facecolor='red', **opz2)
    ax2.set_xlim(-clip, clip)
    ax2.set_xlabel('zero incidence angle')

    ax3.plot(synt1, model_z, **opz1)
    ax3.fill_betweenx(model_z, 0, synt1, where=synt1 > 0, facecolor='blue', **opz2)
    ax3.fill_betweenx(model_z, 0, synt1, where=synt1 < 0, facecolor='red', **opz2)
    ax3.set_xlim(-clip, clip)
    ax3.set_xlabel('angle={:.0f}'.format(ang[-1]))

    ax4.plot(ang, avo, **opz1)
    ax4.set_title('AVO Curve', pad=10)
    ax4.axhline(0, color='k', lw=2)
    ax4.set_xlabel('angle of incidence')
    ax4.set_ylabel('RC', rotation=0, labelpad=10)
    ax4.tick_params(which='major', labelsize=9)

    for aa in [ax0, ax1, ax2, ax3]:
        aa.set_ylim(0, n_samples)
        aa.tick_params(which='major', labelsize=8)
    for aa in [ax1, ax2, ax3]:
        aa.set_yticklabels([])

    plt.subplots_adjust(wspace=1, left=0.03, right=0.95)
