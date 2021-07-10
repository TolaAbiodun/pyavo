import impedance as imp
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from pandas import Series
from numpy import ndarray
from approx import *
import las


def plot_imp(vpvs: Union[ndarray, Series, float], vp: Union[ndarray, Series, float], vs: Union[ndarray, Series, float],
             rho: Union[ndarray, Series, float], angle: Union[float, int], h_well: Union[ndarray, Series],
             h_ref: Union[int, float]):
    """
    Creates log plots of Poisson ratio, Acoustic impedance-Normalized Elastic Impedance(AI-NEI) and Lame's Parameters(lambda-rho & mu-rho)

    :param vpvs: Vp/Vs values form VP/VS Log (m/s)
    :param vp: P-wave velocities form Vp log (m/s)
    :param vs: S-wave velocities from Vs Log (m/s)
    :param rho: Density form RHOB log (g/cc)
    :param angle: Angle of incidence (deg)
    :param h_well: Depth of well from Depth log (m)
    :param h_ref: closest location of the top on the depth array as reference to normal elastic impedance
    :return: Poisson ratio, Acoustic impedance, lambda_rho, mu_rho, NEI
    """

    # poisson ratio
    pr = 0.5 * ((vpvs ** 2 - 2) / (vpvs ** 2 - 1))

    # acoustic impedance
    ai = imp.ai(vp, rho)

    # lambda rho and mu rho
    lambda_rho, mu_rho = imp.lame(vp, vs, rho)

    # normalized elastic impedance
    indx = (np.abs(h_well - h_ref)).argmin()
    nei = imp.norm_ei(vp, vs, rho, vp[indx], vs[indx], rho[indx], angle)

    # Pass the elastic parameters, header to a list
    logs = [pr, ai, nei, lambda_rho, mu_rho]
    header = ['Poisson ratio', 'AI - NEI', r'$\lambda\rho - \mu\rho$']

    # Plot paramaters
    f, ax = plt.subplots(1, 3, figsize=(10, 8))
    for i in range(3):
        ax[i].grid(linestyle='--')
        ax[i].set_ylim(max(h_well), min(h_well))
        if i == 0:
            ax[i].plot(logs[i], h_well, '-', color='black')
            ax[i].set_title(header[i])
        elif i == 1:
            ax[i].plot(logs[1], h_well, '-r', label='AI')
            ax[i].plot(logs[2], h_well, '-g', label='NEI')
            ax[i].set_yticklabels([])
            ax[i].set_title(header[i])
            ax[i].legend(loc='upper right')
        else:
            ax[i].plot(logs[3], h_well, '-b', label=r'$\lambda\rho$')
            ax[i].plot(logs[4], h_well, '-r', label=r'$\mu\rho$')
            ax[i].set_yticklabels([])
            ax[i].set_title(header[i])
            ax[i].legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    return {'Poisson ratio': pr, 'AI': ai, 'NEI': nei, 'lambda_rho': lambda_rho, 'mu_rho': mu_rho}


def crossplot(vp, vs, vpvs, rho, phi, GR, AI, NEI, lambda_rho, mu_rho):
    """
    Generates cross plots of elastic impedance, porosity, density and velocities.

    :param vp: P-wave velocity from Vp log (m/s)
    :param vs: S-wave velocity from Vs log (m/s)
    :param vpvs: VpVS ratio from VP-VS log (m/s)
    :param rho: Density from RHOB log (g/cc)
    :param phi: Porosity from PHI log
    :param GR: Gamma ray from GR log
    :param AI: Acoustic impedance
    :param NEI: Normalized acoustic impedance
    :param lambda_rho: Lame's first parameter
    :param mu_rho: Lame's second parameter
    """
    fig = plt.figure(figsize=(17, 7))

    ax = plt.subplot(2, 4, 1)
    plt.scatter(vp, rho, 20, c=GR, cmap='Spectral')
    ax.set_xlabel('Vp (m/s)')
    ax.set_ylabel('RHOB (g/cc)')
    ax.grid(linestyle='--')
    cbar = plt.colorbar(pad=0)
    cbar.set_label('Gr', labelpad=-19, y=-0.04, rotation=0)

    ax = plt.subplot(2, 4, 2)
    plt.scatter(vp, phi, 20, c=GR, cmap='Spectral')
    ax.set_xlabel('Vp (m/s)')
    ax.set_ylabel('PHI (%)')
    plt.grid()
    cbar = plt.colorbar(pad=0)
    cbar.set_label('GR', labelpad=-19, y=-0.04, rotation=0)

    ax = plt.subplot(2, 4, 3)
    plt.scatter(vs, vpvs, 20, c=GR, cmap='Spectral')
    ax.set_xlabel('Vs (km/s)')
    ax.set_ylabel('Vp/Vs')
    plt.grid()
    cbar = plt.colorbar(pad=0)
    cbar.set_label('GR', labelpad=-19, y=-0.04, rotation=0)

    ax = plt.subplot(2, 4, 4)
    plt.scatter(w5vp, w5vs, 20, c=w5gr, cmap='Spectral')
    ax.set_xlabel('Vp (m/s)')
    ax.set_ylabel('Vs (m/s)')
    plt.grid()
    cbar = plt.colorbar(pad=0)
    cbar.set_label('GR', labelpad=-19, y=-0.04, rotation=0)

    ax = plt.subplot(2, 4, 5)
    plt.scatter(AI, vpvs, 20, c=GR, cmap='Spectral')
    ax.set_xlabel('AI (g/cc x m/s)')
    ax.set_ylabel('Vp/Vs')
    plt.grid()
    cbar = plt.colorbar(pad=0)
    cbar.set_label('GR', labelpad=-19, y=-0.04, rotation=0)

    ax = plt.subplot(2, 4, 6)
    plt.scatter(NEI, vpvs, 20, c=GR, cmap='Spectral')
    ax.set_xlabel('NEI (g/cc x m/s)')
    ax.set_ylabel('Vp/Vs')
    plt.grid()
    cbar = plt.colorbar(pad=0)
    cbar.set_label('GR', labelpad=-19, y=-0.04, rotation=0)

    ax = plt.subplot(2, 4, 7)
    plt.scatter(NEI, AI, 20, c=GR, cmap='Spectral')
    ax.set_xlabel('NEI (g/cc x m/s)')
    ax.set_ylabel('AI (g/cc x m/s)')
    plt.grid()
    cbar = plt.colorbar(pad=0)
    cbar.set_label('GR', labelpad=-19, y=-0.04, rotation=0)

    ax = plt.subplot(2, 4, 8)
    plt.scatter(lambda_rho, mu_rho, 20, c=GR, cmap='Spectral')
    ax.set_xlabel(r'$\lambda\rho (g^2/cc^2 x m^2/s^2)$')
    ax.set_ylabel(r'$\mu\rho (g^2/cc^2 x m^2/s^2)$')
    plt.grid()
    cbar = plt.colorbar(pad=0)
    cbar.set_label('GR', labelpad=-19, y=-0.04, rotation=0)

    plt.tight_layout()
    plt.show()


def avo_plot(angle: ndarray, shuey: float, intercept: float, gradient: float, lim=1.5):
    """
    Generates crossplots of P-wave reflectivity vs angles, & gradient vs intercept.

    :param angle: Angle of incidence
    :param shuey: Shuey 2-term approximation of reflectivity
    :param gradient: Gradient from shuey approx
    :param intercept: Intercept from shuey approx
    """
    #Create subplots
    f, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(angle, shuey, '-', color='r', linewidth=3)
    ax[0].axhline(0, color='k')
    ax[0].set_xlabel('angle ($\\theta$)', fontsize=13)
    ax[0].set_ylabel('R($\\theta$)', fontsize=13, rotation=0, labelpad=13)
    ax[0].set_xlim(0., np.max(angle - 1))
    ax[0].tick_params(labelsize=10)
    ax[0].set_ylim(-lim, lim)
    ax[0].grid(linestyle='--')
    y = np.linspace(-lim, lim, 5)
    ax[0].set_yticks(y)

    ax[1].plot(intercept, gradient, 'o', color='red', markersize=8)
    ax[1].axhline(0, color='k', lw=1)
    ax[1].axvline(0, color='k', lw=1)
    ax[1].set_xlabel('intercept', fontsize=13)
    ax[1].set_ylabel('gradient', fontsize=13, rotation=0, labelpad=13)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    y = np.linspace(-lim, lim, 6)
    x = np.linspace(-lim, lim, 6)
    ax[1].set_yticks(y)
    ax[1].set_xticks(x)
    ax[1].xaxis.set_label_position('bottom')
    ax[1].xaxis.tick_bottom()
    ax[1].yaxis.set_label_position('right')
    ax[1].yaxis.tick_right()
    ax[1].tick_params(labelsize=13)

    plt.tight_layout()
    plt.show()

#Data
well_5 = las.LASReader('well_5.las', null_subs=np.nan)
w5z = well_5.data['DEPT']
w5vp = well_5.data['Vp'] / 100
w5vs = well_5.data['Vs'] / 100
w5vpvs = well_5.data['Vp'] / well_5.data['Vs']
w5rho = well_5.data['RHOB']
w5gr = well_5.data['GR']

rho_mineral = 2.65
rho_fluid = 1.05
w5phi = (rho_mineral - w5rho) / (rho_mineral - rho_fluid)
#
# im = plot_imp(vpvs=w5vpvs, vp=w5vp, vs=w5vs, rho=w5rho, angle=30, h_well=w5z, h_ref=2170)
#
# crossplot(vp=w5vp, vs=w5vs, vpvs=w5vpvs, rho=w5rho, phi=w5phi, GR=w5gr,
#           AI=im['AI'], NEI=im['NEI'], lambda_rho=im['lambda_rho'], mu_rho=im['mu_rho'])

angle = np.arange(0, 31, 1)
#get the closest location of the top on the depth array as reference to NEI
index = (np.abs(w5z - 2170)).argmin()
intercept, gradient, shuey, _ = shuey(w5vp[index], w5vs[index], w5rho[index],
                                       w5vp[index+1], w5vs[index+1],
                                       w5rho[index+1], angle)

avo_plot(angle, shuey, intercept, gradient)
