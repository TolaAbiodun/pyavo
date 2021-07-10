# -*- coding: utf-8 -*-
"""
Functions to convert between various acoustic/elastic parameters,
and provides a way to calculate all the elastic moduli from Vp, Vs, and rho.

Created by Matt Hall, 2014. modified by: Tola Abiodun, 2021

Using equations http://www.subsurfwiki.org/wiki/Elastic_modulus
from Mavko, G, T Mukerji and J Dvorkin (2003), The Rock Physics Handbook, Cambridge University Press
"""

#import libraries
import numpy as np


def youngs(vp=None, vs=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
           pmod=None):
    """
      Computes Young's modulus given either P-wave Velocity, S-wave velocity, and Density, or
    any two elastic moduli (e.g. lambda and mu, or bulk and P moduli).
    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, bulk, pr, and pmod

    :return: Young's modulus in pascals, Pa
    """


    if (vp!=None) and (vs!=None) and (rho!=None):
        return rho * vs**2 * (3.*vp**2 - 4.*vs**2) / (vp**2 - vs**2)

    elif (mu is not None) and (lam is not None):
        return mu * (3.*lam + 2*mu) / (lam + mu)

    elif (bulk is not None) and (lam is not None):
        return 9.*bulk * (bulk - lam) / (3.*bulk - lam)

    elif (bulk is not None) and (mu is not None):
        return 9.*bulk*mu / (3.*bulk + mu)

    elif (lam is not None) and (pr is not None):
        return lam * (1+pr) * (1 - 2*pr) / pr

    elif (pr is not None) and (mu is not None):
        return 2. * mu * (1+pr)

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * (1 - 2*pr)

    else:
        return None


def bulk(vp=None, vs=None, rho=None, mu=None, lam=None, youngs=None, pr=None,
         pmod=None):
    """
     Calculate the Bulk modulus given either P-wave Velocity, S-wave velocity, and Density, or
    any two elastic moduli (e.g. lambda and mu, or bulk and P moduli).
    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, pr, and pmod

    Returns:
        Bulk modulus in pascals, Pa
    """
    if (vp is not None) and (vs is not None) and (rho is not None):
        return rho * (vp**2 - (4./3.)*(vs**2))

    elif (mu is not None) and (lam is not None):
        return lam + 2*mu/3.

    elif (mu is not None) and (youngs is not None):
        return youngs * mu / (9.*mu - 3.*youngs)

    elif (lam is not None) and (pr is not None):
        return lam * (1+pr) / 3.*pr

    elif (pr is not None) and (mu is not None):
        return 2. * mu * (1+pr) / (3. - 6.*pr)

    elif (pr is not None) and (youngs is not None):
        return youngs / (3. - 6.*pr)

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return 1/6. * (3*lam + youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None


def poisson(vp=None, vs=None, rho=None, mu=None, lam=None, youngs=None, bulk=None,
       pmod=None):
    """
     Calculate the Poisson ratio given either P-wave Velocity, S-wave velocity, and Density, or
    any two elastic moduli (e.g. lambda and mu, or bulk and P moduli).
    SI units only.
    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, bulk, and pmod

    Returns:
        Poisson's ratio, dimensionless
    """

    if (vp is not None) and (vs is not None):
        return (vp**2. - 2.*vs**2) / (2. * (vp**2 - vs**2))

    elif (mu is not None) and (lam is not None):
        return lam / (2. * (lam+mu))

    elif (mu is not None) and (youngs is not None):
        return (youngs / (2.*mu)) - 1

    elif (lam is not None) and (bulk is not None):
        return lam / (3.*bulk - lam)

    elif (bulk is not None) and (mu is not None):
        return (3.*bulk - 2*mu) / (6.*bulk + 2*mu)

    elif (bulk is not None) and (youngs is not None):
        return (3.*bulk - youngs) / (6.*bulk)

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return (1/(4*lam)) * (-1*lam - youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None


def mu(vp=None, vs=None, rho=None, pr=None, lam=None, youngs=None, bulk=None,
       pmod=None):
    """
    Computes shear modulus given either Vp, Vs, and rho, or
    any two elastic moduli (e.g. lambda and bulk, or Young's
    and P moduli).
    SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, bulk, youngs, pr, and pmod

    Returns:
        Shear modulus in pascals, Pa
    """

    if (vs is not None) and (rho is not None):
        return rho * vs**2

    elif (bulk is not None) and (lam is not None):
        return 3. * (bulk - lam) / 2.

    elif (bulk is not None) and (youngs is not None):
        return 3. * bulk * youngs / (9.*bulk - youngs)

    elif (lam is not None) and (pr is not None):
        return lam * (1 - 2.*pr) / (2.*pr)

    elif (pr is not None) and (youngs is not None):
        return youngs / (2. * (1 + pr))

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * (1 - 2*pr) / (2. * (1 + pr))

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return 1/4. * (-3*lam + youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None


def lame_param(vp=None, vs=None, rho=None, pr=None,  mu=None, youngs=None, bulk=None,
        pmod=None):
    """
    Computes Lame's first parameter given either P-wave velocity, S-wave velocity, and Density, or
    any two elastic moduli (e.g. bulk and mu, or Young's and P moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from bulk, mu, youngs, pr, and pmod

    Returns:
        Lambda in pascals, Pa
    """
    if (vp is not None) and (vs is not None) and (rho is not None):
        return rho * (vp**2 - 2.*vs**2.)

    elif (youngs is not None) and (mu is not None):
        return mu * (youngs - 2.*mu) / (3.*mu - youngs)

    elif (bulk is not None) and (mu is not None):
        return bulk - (2.*mu/3.)

    elif (bulk is not None) and (youngs is not None):
        return 3. * bulk * (3*bulk - youngs) / (9*bulk - youngs)

    elif (pr is not None) and (mu is not None):
        return 2. * pr * mu / (1 - 2.*pr)

    elif (pr is not None) and (youngs is not None):
        return pr * youngs / ((1+pr) * (1-2*pr))

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * pr / (1+pr)

    else:
        return None


def p_mod(vp=None, vs=None, rho=None, pr=None, mu=None, lam=None, youngs=None,
         bulk=None):
    """
    Computes P-wave modulus given either P-wave velocity, S-wave velocity, and Density, or
    any two elastic moduli (e.g. lambda and mu, or Young's and bulk moduli). SI units only.

    Args:
        vp, vs, and rho
        or any 2 from lam, mu, youngs, pr, and bulk

    Returns:
        P-wave modulus in pascals, Pa
    """

    if (vp is not None) and (rho is not None):
        return rho * vp**2

    elif (lam is not None) and (mu is not None):
        return lam + 2*mu

    elif (youngs is not None) and (mu is not None):
        return mu * (4.*mu - youngs) / (3.*mu - youngs)

    elif (bulk is not None) and (lam is not None):
        return 3*bulk - 2.*lam

    elif (bulk is not None) and (mu is not None):
        return bulk + (4.*mu/3.)

    elif (bulk is not None) and (youngs is not None):
        return 3. * bulk * (3*bulk + youngs) / (9*bulk - youngs)

    elif (lam is not None) and (pr is not None):
        return lam * (1 - pr) / pr

    elif (pr is not None) and (mu is not None):
        return 2. * pr * mu * (1-pr) / (1 - 2.*pr)

    elif (pr is not None) and (youngs is not None):
        return (1-pr) * youngs / ((1+pr) * (1 - 2.*pr))

    elif (pr is not None) and (bulk is not None):
        return 3. * bulk * (1-pr) / (1+pr)

    elif (lam is not None) and (youngs is not None):
        # Note that this returns a tuple.
        x = np.sqrt(9*lam**2 + 2*youngs*lam + youngs**2)

        def b(y): return 1/2. * (-1*lam + youngs + y)

        # Strictly, we should return b(x), b(-x)
        # But actually, the answer is:
        return b(x)

    else:
        return None


def comp_vel(youngs=None, vs=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
       pmod=None):
    """
    Calculates the P-wave velocity given bulk density and any two elastic moduli
    (e.g. lambda and mu, or Young's and P moduli). SI units only.

    Args:
        Any 2 from lam, mu, youngs, pr, pmod, bulk and Rho

    Returns:
        Vp in m/s
    """

    if (mu is not None) and (lam is not None) and (rho is not None):
        return np.sqrt((lam + 2.*mu) / rho)

    elif (youngs is not None) and (mu and rho is not None):
        return np.sqrt(mu * (youngs - 4.*mu) / (rho * (youngs - 3.*mu)))

    elif (youngs is not None) and (pr and rho is not None):
        return np.sqrt(youngs * (1 - pr) / (rho * (1+pr) * (1 - 2.*pr)))

    elif (bulk is not None) and (lam and rho is not None):
        return np.sqrt((9.*bulk - 2.*lam) / rho)

    elif (bulk is not None) and (mu and rho is not None):
        return np.sqrt((bulk + 4.*mu/3.) / rho)

    elif (lam is not None) and (pr and rho is not None):
        return np.sqrt(lam * (1. - pr) / (pr*rho))

    else:
        return None


def shear_vel(youngs=None, vp=None, rho=None, mu=None, lam=None, bulk=None, pr=None,
       pmod=None):
    """
    Computes the Shear wave velocity given bulk density and shear modulus. SI units only.

    Args:
        Mu
        Rho

    Returns:
        Vs in m/s
    """

    if (mu is not None) and (rho is not None):
        return np.sqrt(mu / rho)

    else:
        return None


def elastic_mod(vp, vs, rho):
    """
    Computes elastic moduli given P-wave velocity, S-wave velocity, and Density. SI units only.

    Args:
        Vp, Vs, and rho

    Returns:
        A dict of elastic moduli, plus P-wave impedance.
    """

    mod = {}

    mod['Impedance (rayl)'] = vp * rho

    mod['Shear Modulus (Gpa)'] = round(mu(vs=vs, rho=rho)/1e6, 2)
    mod['Poisson'] = round(poisson(vp=vp, vs=vs, rho=rho), 2)
    mod['Lambda'] = round(lame_param(vp=vp, vs=vs, rho=rho), 2)
    mod['Bulk Modulus (GPa)'] = round(bulk(vp=vp, vs=vs, rho=rho)/1e6, 2)
    mod['P-wave Moduli (GPa)'] = round(p_mod(vp=vp, rho=rho)/1e6, 2)
    mod['Youngs Modulus (GPa)'] = round(youngs(vp=vp, vs=vs, rho=rho)/1e6, 2)

    return mod
