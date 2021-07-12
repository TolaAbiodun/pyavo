# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:51:37 2021

@author: Tola Abiodun
Fluxgate Technologies, NG
References: Wang(2001), Batzle and Wang(1992), Geophysics

Class for modelling fluid properties for gas, oil and brine saturated sands
"""

import warnings
import math
from typing import Union

warnings.filterwarnings("ignore")


def k_rho_matrix(v_cly: float, k_cly: float, k_qtz: float, rho_cly: float, rho_qtz: float) -> tuple:
    """
    Calculate the Bulk modulus and Density of rock matrix.

    :param v_cly: Volume of clay assumed to be 70% of Shale volume
    :param k_cly: Bulk modulus of clay (Gpa)
    :param k_qtz: Bulk modulus of quartz (Gpa)
    :param rho_cly: Density of clay (g/cc)
    :param rho_qtz: Density of quartz (g/cc)
    :returns:
        k_mat : Bulk modulus of rock matrix
        rho_mat : Density of rock matrix
    """
    if isinstance(v_cly, str):
        raise NameError(f'{v_cly} is not a valid data type.\n use integer or float')

    v_qtz = 1 - v_cly
    k_voigt = v_cly * k_cly + v_qtz * k_qtz
    k_reuss = 1 / (v_cly / k_cly + v_qtz / k_qtz)
    k_mat = 0.5 * (k_voigt + k_reuss)
    rho_mat = v_cly * rho_cly + v_qtz * rho_qtz
    # print("The Bulk modulus(Gpa) and Density(g/cc) of the Matrix:")

    return k_mat, rho_mat


def vel_sat(k_sat: float, rho_sat: float, mu: float) -> tuple:
    """
    Estimate the seismic velocities after Gassmann fluid substituion using density and
    elastic moduli of saturated rock.

    :returns:
        vp_new : P-wave velocity
        vs_new : S-wave velocity
    """
    f = 3280.84
    vp_new = math.sqrt((k_sat + (mu * 4 / 3)) / rho_sat) * f
    vs_new = math.sqrt(mu / rho_sat) * f

    return round(vp_new, 2), round(vs_new, 2)


class GassmannSub(object):
    """
    Class to model Gassmann fluid substitution for brine sands, oil sands, and gas sands.
    it generates the P and S wave velocities and density after fluid substitution according to input parameters.

    Arguments
    -----------
    vp = P-wave velocity from log (ft/s)
    vs = S-wave velocity from log (ft/s)
    rho = Bulk density form log (g/cc)
    rho_o = Oil gravity (deg API)
    rho_g = Gas gravity (API)
    vsh = Shale volume from log
    phi = Porosity
    swi = Initial water saturation from log
    swt =  Target water saturation
    S = Salinity (ppm)
    T = Temperature (deg)
    P = Pressure (psi)
    init_fluid = Fluid type of initial hydrocarbon (gas or oil)
    final_fluid = Fluid type of desired output where (gas or oil)
    GOR = Gas-Oil ratio
    """

    def __init__(self, vp: Union[int, float], vs: Union[int, float], rho: Union[int, float], rho_o: Union[int, float],
                 rho_g: Union[int, float], vsh: float, phi: float, swi: float, swt: float, S: float,
                 T: Union[int, float], P: Union[int, float], init_fluid: str, final_fluid: str, GOR: float):
        try:
            self.vp = vp
            self.vs = vs
            self.rho = rho
            self.rho_o = rho_o
            self.rho_g = rho_g
            self.vsh = vsh
            self.phi = phi
            self.swi = swi
            self.swt = swt
            self.S = S
            self.T = T
            self.P = P
            self.init_fluid = init_fluid
            self.final_fluid = final_fluid
            self.GOR = GOR

        except ValueError as err:
            print(f'Input right format {err}')

        except TypeError as err:
            print(f'Input right format {err}')

    def k_rho_brine(self) -> tuple:
        """
        Estimate the bulk modulus and density of brine.

        :returns: k_brine, rho_brine
        """
        # Coefficients for water velocity computation (Batzle and Wang, 1992)
        w11 = 1402.85
        w21 = 4.871
        w31 = -0.04783
        w41 = 1.487e-4
        w51 = -2.197e-7
        w12 = 1.524
        w22 = -0.0111
        w32 = 2.747e-4
        w42 = -6503e-7
        w52 = 7.987e-10
        w13 = 3.437e-3
        w23 = 1.739e-4
        w33 = -2.135e-6
        w43 = -1.455e-8
        w53 = 5.23e-11
        w14 = -1.197e-5
        w24 = -1.628e-6
        w34 = 1.237e-8
        w44 = 1.327e-10
        w54 = -4.614e-13

        # Create a dictionary for the constants
        constants_dict = {
            'w11': w11,
            'w21': w21,
            'w31': w31,
            'w41': w41,
            'w51': w51,
            'w12': w12,
            'w22': w22,
            'w32': w32,
            'w42': w42,
            'w52': w52,
            'w13': w13,
            'w23': w23,
            'w33': w33,
            'w43': w43,
            'w53': w53,
            'w14': w14,
            'w24': w24,
            'w34': w34,
            'w44': w44,
            'w54': w54
        }
        # Convert Pressure(psi) to Mpa
        P = self.P * 6.894757 * 0.001
        # Express salinity(ppm) as weight fraction
        S = self.S * 1e-6

        vw = 0
        for i in range(1, 6):
            for j in range(1, 5):
                constant_key = 'w' + str(i) + str(j)
                constant = constants_dict[constant_key]
            vw += constant * (self.T ** (i - 1)) * (P ** (j - 1))

        v1 = 1170 - 9.6 * self.T + 0.055 * self.T * self.T - 8.5 * 10 ** (
            -5) * self.T * self.T * self.T + 2.6 * P - (0.0029 * self.T * P) - (0.0476 * P ** 2)
        v_brine = vw + S * v1 + S ** 1.5 * (780 - 10 * P + 0.16 * P * P) - 1820 * S * S
        r1 = 489 * P - 2 * self.T * P + 0.016 * self.T * self.T * P - 1.3 * 10 ** (
            -5) * self.T * self.T * self.T * P - 0.333 * P * P - 0.002 * self.T * P * P
        rho_water = 1 + 10 ** (-6) * (-80 * self.T - 3.3 * self.T * self.T + 0.00175 * self.T * self.T * self.T + r1)
        r2 = 300 * P - 2400 * P * S + self.T * (80 + 3 * self.T - 3300 * S - 13 * P + 47 * P * S)
        rho_brine = rho_water + 0.668 * S + 0.44 * S * S + 10 ** (-6) * S * r2
        k_brine = rho_brine * v_brine ** 2 * 1e-6

        return k_brine, rho_brine

    # Function to estimate initial hydrocarbon properties
    def init_hyc(self) -> tuple:
        """
        Estimate Bulk modulus and density of initial hydrocarbon.

        :return: k_hyc: Bulk modulus
                 rho_hyc: Density
        """
        rho_hyc = 0
        k_hyc = 0
        if self.init_fluid == 'oil':  # Default is oil in this case
            P = self.P * 6.894757 * 0.001  # convert Pressure from Psi to Mpa
            rho_o = 141.5 / (self.rho_o + 131.5)
            div_mill = 1 / 1000000
            Bo = 0.972 + 0.00038 * (2.495 * self.GOR * math.sqrt(self.rho_g / rho_o) + self.T + 17.8) ** 1.175
            rho_p = rho_o / ((1 + 0.001 * self.GOR) * Bo)
            rho_s = (rho_o + 0.0012 * self.GOR * self.rho_g) / Bo
            r = rho_s + (0.00277 * P - 1.71 * 0.0000001 * P * P * P) * (rho_s - 1.15) ** 2 + (3.49 * 0.0001 * P)
            rho_hyc += r / (0.972 + 3.81 * 0.0001 * (self.T + 17.78) ** 1.175)
            y = math.sqrt(18.33 / rho_p - 16.97)
            vel = 2096 * math.sqrt(rho_p / (2.6 - rho_p)) - 3.7 * self.T + 4.64 * P + 0.0115 * (y - 1) * self.T * P
            k_hyc += rho_hyc * vel * vel * div_mill
            # print('Bulk modulus(Gpa) and Density(g/cc) of initial fluid (oil)')
        elif self.init_fluid == 'gas':
            # Only gas is present in the reservoir. A gas sand case
            k_hyc = 0
            rho_hyc = 0
            R = 8.314
            P = self.P * 6.894757 * 0.001  # convert Pressure from Psi to Mpa
            Tabs = self.T + 273.15
            Ppr = P / (4.892 - 0.4048 * self.rho_g)
            Tpr = Tabs / (94.72 + 170.75 * self.rho_g)
            E1 = math.exp(-Ppr ** 1.2 / Tpr * (0.45 + 8 * (0.56 - 1 / Tpr) ** 2))
            E = 0.109 * (3.85 - Tpr) ** 2 * E1
            Z1 = 0.03 + 0.00527 * (3.5 - Tpr) ** 3
            Z = Z1 * Ppr + 0.642 * Tpr - 0.007 * Tpr ** 4 - 0.52 + E
            rho_hyc += 28.8 * self.rho_g * P / (Z * R * Tabs)
            F = -1.2 * Ppr ** 0.2 / Tpr * (0.45 + 8 * (0.56 - 1 / Tpr)) * E1
            dz_dp = Z1 + 0.109 * (3.85 - Tpr) ** 2 * F
            yo = 0.85 + 5.6 / (Ppr + 2) + 27.1 / (Ppr + 3.5) ** 2 - (8.7 * math.exp(-0.65 * (Ppr + 1)))
            k_hyc += P / (1 - (Ppr / Z * dz_dp)) * yo / 1000
            # print('Bulk modulus(GPa) and Density(g/cc) of initial fluid (gas)')
        return k_hyc, rho_hyc

    def k_rho_fluid(self) -> tuple:
        """
        Estimate the Bulk modulus and density of the mixed pore fluid phase (Initial insitu model)

        :return: k_fld: Bulk modulus of pore fluid
                 rho_fld: Density of pore fluid
        """
        k_hyc, rho_hyc = self.init_hyc()
        k_br, rho_br = self.k_rho_brine()
        shi = 1 - self.swi
        k_fl = (k_br / self.swi + k_hyc / shi)
        rho_fl = self.swi * rho_br + shi * rho_hyc
        return k_fl, rho_fl

    def insitu_moduli(self, rho_fluid: float, rho_matrix: float, d_phi=True) -> tuple:
        """
        Estimate the initial original moduli for saturated insitu rock.
        Density of the insitu saturated rock is calculated from the porosity log using the mass balance equation.

        :param rho_fluid: density of insitu pore fluid phase (g/cc)
        :param rho_matrix: density of rock matrix (g/cc)
        :param d_phi: density derived from input RHOB log.
        :return: k_sat_init: Bulk modulus
                 mu_sat_init: Shear modulus
        """
        # Use porosity to estimate initial saturated rock density
        factor = 0.000305
        vp = self.vp * factor
        vs = self.vs * factor

        init_rho = self.phi * rho_fluid + (1 - self.phi) * rho_matrix
        if d_phi:  # Use density from RHOB log
            k_sat_init = self.rho * (vp ** 2 - (vs ** 2 * 4 / 3))
            mu_sat_init = self.rho * vs * vs
        else:  # use calculated density from porosity
            k_sat_init = init_rho * (vp ** 2 - (vs ** 2 * 4 / 3))
            mu_sat_init = init_rho * vs * vs

        return k_sat_init, mu_sat_init

    def k_frame(self, k_mat: float, k_fld: float, k_sat: float) -> float:
        """
        Estimate the bulk modulus of porous rock frame.

        :param k_mat: Bulk modulus of rock matrix
        :param k_fld: Bulk modulus of pore fluid phase
        :param k_sat: Bulk modulus of saturated insitu rock
        :return: k_frame: Bulk modulus
        """
        k0 = k_sat * (self.phi * k_mat / k_fld + 1 - self.phi) - k_mat
        k1 = (self.phi * k_mat / k_fld) + (k_sat / k_mat) - 1 - self.phi
        k_frame = k0 / k1
        return k_frame

    def final_hc(self) -> tuple:
        """
        Estimate the bulk modulus and density of the desired hydrocarbon (oil or gas)

        :return: k_hyc: bulk modulus, rho_hyc: density
        """
        # Output fluid is brine

        if self.final_fluid == 'brine':
            # print('Bulk modulus(Gpa) and Density(g/cc) of desired fluid (brine)')
            k_hyc, rho_hyc = self.k_rho_brine()

        # Output fluid is Oil
        elif self.final_fluid == 'oil':
            k_hyc = 0
            rho_hyc = 0
            P = self.P * 6.894757 * 0.001  # convert Pressure from Psi to Mpa
            rho_o = 141.5 / (self.rho_o + 131.5)
            factor = 1 / 1000000
            Bo = 0.972 + 0.00038 * (2.495 * self.GOR * math.sqrt(self.rho_g / rho_o) + self.T + 17.8) ** 1.175
            rho_p = rho_o / ((1 + 0.001 * self.GOR) * Bo)
            rho_s = (rho_o + 0.0012 * self.GOR * self.rho_g) / Bo
            r = rho_s + (0.00277 * P - 1.71 * 0.0000001 * P * P * P) * (rho_s - 1.15) ** 2 + (3.49 * 0.0001 * P)
            rho_hyc += r / (0.972 + 3.81 * 0.0001 * (self.T + 17.78) ** 1.175)
            y = math.sqrt(18.33 / rho_p - 16.97)
            vel = 2096 * math.sqrt(rho_p / (2.6 - rho_p)) - 3.7 * self.T + 4.64 * P + 0.0115 * (y - 1) * self.T * P
            k_hyc += rho_hyc * vel * vel * factor
            # print('Bulk modulus(Gpa) and Density(g/cc) of desired fluid (oil)')

        # Output fluid is Gas
        elif self.final_fluid == 'gas':
            k_hyc = 0
            rho_hyc = 0
            R = 8.314
            P = self.P * 6.894757 * 0.001  # convert Pressure from Psi to Mpa
            Tabs = self.T + 273.15
            Ppr = P / (4.892 - 0.4048 * self.rho_g)
            Tpr = Tabs / (94.72 + 170.75 * self.rho_g)
            E1 = math.exp(-Ppr ** 1.2 / Tpr * (0.45 + 8 * (0.56 - 1 / Tpr) ** 2))
            E = 0.109 * (3.85 - Tpr) ** 2 * E1
            Z1 = 0.03 + 0.00527 * (3.5 - Tpr) ** 3
            Z = Z1 * Ppr + 0.642 * Tpr - 0.007 * Tpr ** 4 - 0.52 + E
            rho_hyc += 28.8 * self.rho_g * P / (Z * R * Tabs)
            F = -1.2 * Ppr ** 0.2 / Tpr * (0.45 + 8 * (0.56 - 1 / Tpr)) * E1
            dz_dp = Z1 + 0.109 * (3.85 - Tpr) ** 2 * F
            yo = 0.85 + 5.6 / (Ppr + 2) + 27.1 / (Ppr + 3.5) ** 2 - (8.7 * math.exp(-0.65 * (Ppr + 1)))
            k_hyc += P / (1 - (Ppr / Z * dz_dp)) * yo / 1000
            # print('Bulk modulus(Gpa) and Density(g/cc) of desired fluid (gas)')
        return k_hyc, rho_hyc

    def k_rho_sat(self, k_mat: float, rho_mat: float, k_frame: float) -> tuple:
        """
        Estimate the Gassmann bulk modulus and density of saturated rock after fluid substitution.
        The desired fluid is defined by the target water saturation and type of hydrocarbon.

        :param k_mat: Bulk modulus of rock matrix (GPa)
        :param rho_mat: Density of rock matrix (g/cc)
        :param k_frame: Bulk modulus of rock frame (Gpa)
        :return: k_sat_new: Bulk modulus
                 rho_sat_new: Density
        """
        # Calculate density of saturated rock
        k_brine, rho_brine = self.k_rho_brine()
        k_hyc, rho_hyc = self.final_hc()
        sht = 1 - self.swt
        k_fld = 1 / (self.swt / k_brine + sht / k_hyc)
        rho_fld = self.swt * rho_brine + sht * rho_hyc
        rho_sat_new = self.phi * rho_fld + (1 - self.phi) * rho_mat
        # print('Saturated rock density(g/cc)')

        # Calculate Gassmann saturated bulk modulus
        k1 = self.phi / k_fld + (1 - self.phi) / k_mat - k_frame / (k_mat * k_mat)
        k_sat_new = k_frame + ((1 - k_frame / k_mat) ** 2) / k1

        return k_sat_new, rho_sat_new
