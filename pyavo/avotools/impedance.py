# -*- coding: utf-8 -*-
# Functions to calculate acoustic and elastic impedances
"""
Created on Thur 1 21:39:03 2021
@author: Tola Abiodun
"""
import numpy as np


def ai(vp: float, rho: float):
    """
    Calculates the acoustic impedance of a layer

    :param vp: P-wave velocity (m/s)
    :param rho: Density (g/cc)
    :returns: Acoustic Impedance : float
    """

    z = vp * rho
    return z


def ei(vp, vs, rho, ang):
    """
    Calculates the elastic impedance of a layer.
    :param vp: P-wave velocity (m/s)
    :param vs: S-wave velocity (m/s)
    :param rho: Density (g/cc)
    :param ang: Angle of incidence (deg)
    :returns: Elastic impedance

    Reference:
    Connolly, P., 1999, Elastic impedance: The Leading Edge, 18, 438–452.
    """
    theta = np.radians(ang)
    k = (vs / vp) ** 2

    x = 1 + np.tan(theta) ** 2
    y = -8 * k * np.sin(theta) ** 2
    z = 1 - 4 * k * np.sin(theta) ** 2

    ei = (vp ** x) * (vs ** y) * (rho ** z)
    return ei


def norm_ei(vp, vs, rho, vp_sh, vs_sh, rho_sh, ang):
    """
    Calculates the normalized elastic impedance.

    :param vp: P-wave velocity. (m/s)
    :param vs: S-wave velocity. (m/s)
    :param rho: Density. (g/cc)
    :param vp_sh: P-wave velocity in shale (m/s)
    :param vs_sh: S-wave velocity in shale (m/s)
    :param rho_sh: Shale reference density.
    :param ang: Angle of incidence
    :returns: Normalized elastic impedance.

    Reference:
    Whitcombe, D, 2002, Elastic impedance normalization, Geophysics, 67 (1), 60–62.
    """

    theta = np.radians(ang)
    k = (vs / vp) ** 2

    x = 1 + np.tan(theta) ** 2
    y = -8 * k * np.sin(theta) ** 2
    z = 1 - 4 * k * np.sin(theta) ** 2

    nei = vp_sh * rho_sh * ((vp / vp_sh) ** x) * ((vs / vs_sh) ** y) * ((rho / rho_sh) ** z)
    return nei


def lame(vp, vs, rho):
    """
    Computes Lamé parameters - lambda_rho and mu_rho.

    :param vp: P-wave velocity (m/s)
    :param vs: S-wave velocity (m/s)
    :param rho: Density (g/cc)
    :returns: Lamé constants - lambda_rho and mu_rho

    Reference
    ---------
    Goodway, B., T. Chen, and J. Downton, 1997, Improved AVO fluid detection
    and lithology discrimination using Lamé petrophysical parameters; “λρ”,
    “μρ”, & “λ/μ fluid stack” from P and S inversions: 67th Annual
    International Meeting, SEG, Expanded Abstracts, 183-186.
    """

    vp_imp = vp * rho
    vs_imp = vs * rho
    lambda_rho = vp_imp ** 2 - 2 * vs_imp ** 2
    mu_rho = vs_imp ** 2

    return lambda_rho, mu_rho
