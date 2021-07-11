"""
Functions to import segy data and compute AVO attributes from Near and Far Stacks
"""
import segyio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray import DataArray


# Import segy data
def read_segy(filepath: str, byte_il=14, byte_xl=21, ignore_geometry=False) -> DataArray:
    """
    Read a segyfile into a Data array
    :param filepath: filepath, any valid string path is acceptable
    :param byte_il: inline byte
    :param byte_xl: crossline byte
    :return: Data
    """
    with segyio.open(filepath, iline=byte_il, xline=byte_xl, ignore_geometry=ignore_geometry) as f:
        header = segyio.tools.wrap(f.text[0])
        inlines = f.ilines / 1e3
        crosslines = f.xlines
        samp_rate = segyio.tools.dt(f) / 1e3
        n_samp = f.samples.size
        n_traces = f.tracecount
        cube = segyio.tools.cube(f)
        twt = f.samples + crosslines.min()
    file_size = cube.nbytes / 1024 ** 2
    print(f'number of traces: {n_traces}, samples: {n_samp}, sample rate: {samp_rate} s')
    print(f'first, last sample twt: {twt[0]}, {twt[-1]} s')
    print('file_size: {:.2f} Mb ({:.2f} Gb)'.format(file_size, file_size / 1024))
    print(f'inlines: {inlines.size}, min={inlines.min()}, max={inlines.max()}')
    print(f'crosslines: {crosslines.size}, min={crosslines.min()}, max={crosslines.max()}')
    data_array = xr.DataArray(cube, [('IL', inlines), ('XL', crosslines), ('TWT', twt)])

    return data_array


# Helper method
def _read_horizon(filepath: str, labels=('inline', 'crossline', 'depth')):
    return np.recfromtxt(filepath, names=labels)


def nfstack(horizon_data: str, near_stack: DataArray, far_stack: DataArray,
            inline: int, robust=True, interpolation='spline16',
            cmap='RdBu', well_XL=None, well_display=True):
    """
    Plot Near and Far angle stacks from X-DataArray

    :param horizon_data: file path, any valid string path to a txt file is acceptable.
    :param near_stack: DataArray of Near angle stack
    :param far_stack: DataArray of Far angle stack
    :param inline: Inline number
    :param well_XL: Well crossline number to define well position
    :param well_display: Show well trajectory on plot, vertical
    """

    f, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Plot near stack
    hz = _read_horizon(horizon_data)
    near_stack.sel(IL=inline).plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax[0],
                                          robust=robust, interpolation=interpolation, cmap=cmap)
    ax[0].set_title('Near Stack', fontsize=15, fontweight='bold', pad=10)

    # Plot far stack
    far_stack.sel(IL=inline).plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax[1],
                                         robust=robust, interpolation=interpolation, cmap=cmap)
    ax[1].set_title('Far Stack', fontsize=15, fontweight='bold', pad=10)

    for aa in ax:
        aa.plot(hz[hz['inline'] == inline]['crossline'], hz[hz['inline'] == inline]['depth'], color='r', alpha=0.5, lw=4)
        aa.xaxis.set_label_position('top')
        aa.xaxis.tick_top()
        if well_XL is not None and well_display:
            aa.axvline(well_XL, color='k', ls='--', lw=1.5)
            aa.text(well_XL, near_stack['TWT'][-1], 'Well', fontsize=13, fontweight='bold', color='white',
                       bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})
    plt.tight_layout()
    plt.show()


def nf_attributes(horizon_data: str, near_stack: DataArray, far_stack: DataArray, TWT_slice: tuple,
                  XL_slice: tuple, inline: int, well_XL=None, well_display=True, robust=True,
                  interpolation='spline16', add_colorbar=True, cmap='jet_r'):
    """
    Display near and far angle stack attributes.

    Reference: Avseth, P., Mukerji, T., & Mavko, G., 2010.
    Quantitative seismic interpretation: Applying rock physics tools to reduce interpretation risk.Cambridge university press.

    :param horizon_data: file path, any valid string path to a txt file is acceptable.
    :param near_stack: DataArray of Near angle stack
    :param far_stack: DataArray of Far angle stack
    :param TWT_slice: Two-way time slice
    :param XL_slice: Crossline time slice
    :param inline: Inline number
    :param well_XL: Well crossline number to define well position
    :param well_display: Show well trajectory on plot, vertical
    """
    hz = _read_horizon(horizon_data)
    near_il = near_stack.sel(IL=inline)
    far_il = far_stack.sel(IL=inline)
    t0, t1 = TWT_slice[0], TWT_slice[1]
    x0, x1 = XL_slice[0], XL_slice[1]
    near_inline_max = near_il.sel(TWT=slice(t0, t1), XL=slice(x0, x1))
    far_inline_max = far_il.sel(TWT=slice(t0, t1), XL=slice(x0, x1))

    # Compute near and far attributes
    fn_att = far_inline_max - near_inline_max
    fnn_att = (far_inline_max - near_inline_max) * near_inline_max
    fnf_att = (far_inline_max - near_inline_max) * far_inline_max

    # plot attributes
    plot_ppt = {'robust': robust, 'interpolation': interpolation,
                'add_colorbar': add_colorbar, 'cmap': cmap}

    f, ax = plt.subplots(1, 3, figsize=(15, 5))

    fn_att.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax[0], **plot_ppt)
    fnn_att.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax[1], **plot_ppt)
    fnf_att.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax[2], **plot_ppt)
    ax[0].set_title('(Far - Near)', fontweight='bold', fontsize=14, pad=10)
    ax[1].set_title('(Far - Near) x Near', fontweight='bold', fontsize=14, pad=10)
    ax[2].set_title('(Far - Near) x Far', fontweight='bold', fontsize=14, pad=10)

    for aa in ax:
        aa.plot(hz[hz['inline'] == inline]['crossline'], hz[hz['inline'] == inline]['depth'], color='k', lw=3, alpha=0.5)
        aa.set_xlim(x0, x1)
        aa.set_ylim(t1, t0)
        aa.xaxis.set_label_position('top')
        aa.xaxis.tick_top()
        if well_XL is not None and well_display:
            aa.axvline(well_XL, color='k', ls='--', lw=1.5)
            aa.text(well_XL, t1, 'Well', fontsize=12, fontweight='bold', color='white',
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})

    plt.tight_layout()
    plt.show()


def avo_attributes(horizon_data: str, near_stack: DataArray, far_stack: DataArray, theta_near: int, theta_far: int,
                   TWT_slice: tuple, XL_slice: tuple, inline: int, well_XL=None, well_display=True, robust=True,
                   interpolation='spline16', add_colorbar=True, cmap='jet_r') -> dict:
    """
    Display AVO attributes and crossplot.

    :param horizon_data: file path, any valid string path to a txt file is acceptable.
    :param near_stack: DataArray of Near angle stack
    :param far_stack: DataArray of Far angle stack
    :param theta_near: Incidence angle of near stack
    :param theta_far: Incidence angle of far stack
    :param TWT_slice: Two-way time slice
    :param XL_slice: Crossline time slice
    :param inline: Inline number
    :param well_XL: Well crossline number to define well position
    :returns: Dict - AVO intercept and gradient values
    """
    hz = _read_horizon(horizon_data)
    near_il = near_stack.sel(IL=inline)
    far_il = far_stack.sel(IL=inline)
    near_ang = np.sin(np.radians(theta_near)) ** 2
    far_ang = np.sin(np.radians(theta_far)) ** 2

    # calculte intercept(c) and gradient(m)
    m = (far_il - near_il) / (far_ang - near_ang)
    c = near_il - m * near_ang

    # get a slice
    t0, t1 = TWT_slice[0], TWT_slice[1]
    x0, x1 = XL_slice[0], XL_slice[1]
    c_slice = c.sel(TWT=slice(t0, t1), XL=slice(x0, x1))
    m_slice = m.sel(TWT=slice(t0, t1), XL=slice(x0, x1))

    # Return flattened array of gradient and intercept slice
    c_ravel = np.ravel(c_slice)
    m_ravel = np.ravel(m_slice)

    # Fit a 1st order polynomial and get the coefficients
    i, j = np.polynomial.polynomial.polyfit(c_ravel, m_ravel, 1)

    # AVO attributes - fluid, product and RC diff
    avo_prod = c_slice * m_slice
    rc_diff = (c_slice + m_slice) * 0.5
    fluid_fact = c_slice - j * m_slice

    # Plot AVO attributes and crossplot
    f, [[ax0, ax1, ax2], [ax3, ax4, ax5]] = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    ax = [ax0, ax1, ax2, ax3, ax4]

    # plot attributes
    plot_ppt = {'robust': robust, 'interpolation': interpolation,
                'add_colorbar': add_colorbar, 'cmap': cmap}

    avo_prod.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax0, **plot_ppt)
    ax0.set_title('AVO Product', fontweight='bold', fontsize=14, pad=7)

    rc_diff.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax1, **plot_ppt)
    ax1.set_title('Reflection Coefficient Difference', fontweight='bold', fontsize=14, pad=7)

    fluid_fact.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax2, **plot_ppt)
    ax2.set_title('Fluid Factor', fontweight='bold', fontsize=14, pad=7)

    c_slice.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax3, **plot_ppt)
    ax3.set_title('AVO Intercept', fontweight='bold', fontsize=14, pad=7)

    m_slice.plot.imshow(x='XL', y='TWT', yincrease=False, ax=ax4, **plot_ppt)
    ax4.set_title('AVO Gradient', fontweight='bold', fontsize=14, pad=7)

    ax5.plot(c_slice, m_slice, '.r', alpha=0.2)
    ax5.plot(np.ravel(c_slice), i + j * np.ravel(c_slice), '-k')
    ax5.set_title('AVO crossplot (Intercept vs Gradient', fontweight='bold', fontsize=14, pad=7)
    ax5.set_xlabel('Intercept')
    ax5.set_ylabel('Gradient')
    ax5.grid()

    for aa in ax:
        aa.plot(hz[hz['inline'] == inline]['crossline'], hz[hz['inline'] == inline]['depth'], color='k', lw=3, alpha=0.5)
        aa.set_xlim(x0, x1)
        aa.set_ylim(t1, t0)
        aa.xaxis.set_label_position('top')
        aa.xaxis.tick_top()
        if well_XL is not None and well_display:
            aa.axvline(well_XL, color='k', ls='--', lw=1.5)
            aa.text(well_XL, t1, 'Well', fontsize=12, fontweight='bold', color='white',
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})

    plt.tight_layout()
    plt.show()
    return {'Intercept': c, 'Gradient': m}
