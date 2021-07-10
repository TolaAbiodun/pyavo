"""
Functions to compute Aki-Richards and Shuey 2-term approximations
"""

import numpy as np
from numpy import ndarray
from typing import Union
from pandas import Series


def ref_coeff(imp: Union[ndarray, Series]):
    """
    Computes the reflection coefficient for a plane incident P-wave.

    :param imp: Acoustic Impedance
    :returns:
        rc : array
        The reflection coefficient
    """
    rc = (imp[1:] - imp[:-1]) / (imp[1:] + imp[:-1])
    rc = np.append(rc, rc[-1])

    return (rc)


def snell(vp1: ndarray, vp2: ndarray, theta1: float):
    """
    Computes the angles of refraction for an incident P-wave in a two-layered
    model.

    Reference:
    AVO - Chopra and Castagna, 2014, Page 6.

    :param vp1: P-wave velocity in upper layer
    :param vp2: P-wave velocity in lower layer
    :param theta1: Angle of incidence
    :returns: Angle of Refraction, Ray Parameter
    """

    p = np.sin(theta1) / vp1
    theta2 = np.arcsin(p * vp2)

    return (theta2, p)


def shueyrc(vp0: Union[ndarray, Series, float], vs0: Union[ndarray, Series, float],
            rho0: Union[ndarray, Series, float], theta1: Union[ndarray, Series, float]):
    """
    Computes the P-wave reflectivity with Shuey (1985) 2 terms for a
    given well log.
    Reference: Avseth et al., Quantitative seismic interpretation, 2006, Page 182.

    :param vp0: P-wave velocities from Vp log
    :param vs0: S-Wave velocities from Vs log
    :param rho0: Density from RHOB log
    :param theta1: Angles of incidence
    :return:
        RC : array
            Reflection coefficient for the 2-term approximation.
        c : array
            Intercept.
        m : array
            Gradient.
    """

    theta1 = np.radians(theta1)

    dvp = vp0[1:] - vp0[:-1]
    dvs = vs0[1:] - vs0[:-1]
    drho = rho0[1:] - rho0[:-1]

    drho = np.insert(drho, 0, drho[0])
    dvp = np.insert(dvp, 0, dvp[0])
    dvs = np.insert(dvs, 0, dvs[0])

    vp = (vp0[1:] + vp0[:-1]) / 2.0
    vs = (vs0[1:] + vs0[:-1]) / 2.0
    rho = (rho0[1:] + rho0[:-1]) / 2.0

    vp = np.insert(vp, 0, vp[0])
    vs = np.insert(vs, 0, vs[0])
    rho = np.insert(rho, 0, rho[0])

    c = 0.5 * (dvp / vp + drho / rho)
    m = 0.5 * dvp / vp - 2 * (vs ** 2 / vp ** 2) * (drho / rho + 2 * dvs / vs)

    t1 = np.outer(c, 1)
    t2 = np.outer(m, np.sin(theta1) ** 2)

    RC = t1 + t2
    return (RC, c, m)


def aki_richards(vp1: ndarray, vs1: ndarray, rho1: Union[ndarray, float], vp2: ndarray,
                 vs2: ndarray, rho2: Union[ndarray, float], theta1: ndarray) -> ndarray:
    """
    Computes the Reflection Coefficient with Aki and Richard's (1980) equation for
    a two-layered model.
    Reference: AVO - Chopra and Castagna, 2014, Page 62.

    :param vp1: P-wave velocity in upper layer
    :param vs1: S-wave velocity in upper layer
    :param rho1: Density of upper layer
    :param vp2: P-wave velocity in lower layer
    :param vs2: S-wave velocity in lower layer
    :param rho2: Density in lower layer
    :param theta1: Angle of incidence
    :return: Reflection coefficient
    """

    theta1 = np.radians(theta1)
    theta2, p = snell(vp1, vp2, theta1)
    theta = (theta1 + theta2) / 2.

    vp = (vp1 + vp2) / 2
    vs = (vs1 + vs2) / 2
    rho = (rho1 + rho2) / 2

    r1 = 0.5 * (1 - 4 * p ** 2 * vs ** 2) * (rho2 - rho1) / rho
    r2 = 0.5 / (np.cos(theta) ** 2) * (vp2 - vp1) / vp
    r3 = 4 * p ** 2 * vs ** 2 * (vs2 - vs1) / vs

    rc = r1 + r2 - r3

    return (rc)


def shuey(vp1: ndarray, vs1: ndarray, rho1: ndarray,
          vp2: ndarray, vs2: ndarray, rho2: ndarray, theta1: ndarray):
    """
    Computes the Reflectiviy parameters with Shuey (1985) 2 and 3 terms for a
    two-layered model.
    Reference:
    Avseth et al., Quantitative seismic interpretation, 2006, Page 182.

    :param vp1: P-wave velocity in upper layer
    :param vs1: S-wave velocity in lower layer
    :param rho1: Density in upper later
    :param vp2: P-wave velocity in lower layer
    :param vs2: S-wave velocity in lower layer
    :param rho2: Density in lower layer
    :param theta1: Angles of incidence
    :returns:
        c : array
            Intercept.
        m : array
            Gradient.
        rc2 : array
            Reflection coefficient for the 2-term approximation.
        rc3 : array
            Reflection coefficient for the 3-term approximation.
    """

    theta1 = np.radians(theta1)

    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    vp = (vp1 + vp2) / 2
    vs = (vs1 + vs2) / 2
    rho = (rho1 + rho2) / 2

    c = 0.5 * (dvp / vp + drho / rho)
    m = 0.5 * (dvp / vp) - 2 * (vs ** 2 / vp ** 2) * (drho / rho + 2 * (dvs / vs))
    F = 0.5 * (dvp / vp)

    rc2 = c + m * np.sin(theta1) ** 2
    rc3 = c + m * np.sin(theta1) ** 2 + F * (np.tan(theta1) ** 2 - np.sin(theta1) ** 2)

    return (c, m, rc2, rc3)
