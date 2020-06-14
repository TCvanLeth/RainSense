# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:31:25 2015

@author: T.C. van Leth
"""

import numpy as np
import pyhad as ph
from pyhad import ufuncs as uf
import scatterpy as scat
from tqdm import tqdm


def calc_scatter(freq, D, T):
    m = get_m_rain(T, freq)
    wl = ph.constants.light/freq
    shape = dropshape2

    S = get_scatter(D, wl, m, shape)
    dpc = get_dpc(S, wl)
    ext = get_ext(S, wl)
    ref = get_refl(S, wl)
    return ph.Channel([S, ext, ref, dpc])


###############################################################################
def get_refl(S, wl):
    """
    Backscatter reflectivity cross section
    """
    Sh = 2*uf.abs2(S.sel(direction='backward', out_pol='H', in_pol='H'))
    Sv = 2*uf.abs2(S.sel(direction='backward', out_pol='V', in_pol='V'))

    cross = ph.merge([Sh, Sv], xdim='in_pol')
    del cross['out_pol']
    cross.swap_dims(in_pol='polarization')
    cross *= wl**2/np.pi

    cross.setattrs(quantity='backscatter_cross_section', unit='m2')
    cross.setname('refl')
    return cross


def get_ext(S, wl):
    """
    Exctinction cross section
    """
    Sh = uf.imag(S.sel(direction='forward', out_pol='H', in_pol='H'))
    Sv = uf.imag(S.sel(direction='forward', out_pol='V', in_pol='V'))

    cross = ph.merge([Sh, Sv], xdim='in_pol')
    del cross['out_pol']
    cross.swap_dims(in_pol='polarization')
    cross *= wl**2/np.pi

    cross.setattrs(quantity='extinction_cross_section', unit='m2')
    cross.setname('ext')
    return cross


def get_dpc(S, wl):
    """
    Differential phase cross section
    """
    Sh = uf.real(S.sel(direction='forward', out_pol='H', in_pol='H'))
    Sv = uf.real(S.sel(direction='forward', out_pol='V', in_pol='V'))
    cross = (Sh - Sv) * wl**2/np.pi

    cross.setattrs(quantity='differential_phase_cross_section', unit='m2')
    cross.setname('dpc')
    return cross


def get_scatter(D, wl, m, shape):
    """
    Use T-matrix method to compute the scattering amplitude matrix.
    """
    geom = ph.index(['backward', 'forward'], 'direction')
    theta0 = ph.Array([0.5*np.pi, 0.5*np.pi], (geom,))
    theta = ph.Array([np.pi, 0.5*np.pi], (geom,))
    phi0 = ph.Array([0.0, 0.0], (geom,))
    phi = ph.Array([0.0, 0.0], (geom,))

    orient = scat.orientation.sph_gauss_pdf(np.deg2rad(2))
    wl = wl.rechunk({'frequency':1})
    m = m.rechunk({'frequency':1, 'temperature':1})
    D = D.rechunk({'diameter':50})
    S = get_S_oa(D, wl, m, shape, theta0, theta, phi0, phi, orient)
    S = S.rechunk(ph.common.CHUNKSIZE)
    S.setname('scatter')
    return S

def get_shape(D):
    """
    Axis ratio of spheroid hydrometeor of diameter D.
    """
    rain = dropshape2(D)
    rain.setattrs(htype='rain')

    snow = get_shape_snow(D)  # placeholder!
    hail = get_shape_hail(D)  # placeholder!
    mix = get_shape_mix(D)  # placeholder!
    graupel = get_shape_graupel(D)  # placeholder!

    ar = ph.Array.merged([rain, hail, snow, mix, graupel], xdim='htype')
    return ar


def get_shape_snow(D):
    shape = D**0
    shape.setattrs(htype='snow')
    return shape


def get_shape_hail(D):
    shape = D**0
    shape.setattrs(htype='hail')
    return shape


def get_shape_mix(D):
    shape = D**0
    shape.setattrs(htype='mix')
    return shape


def get_shape_graupel(D):
    shape = D**0
    shape.setattrs(htype='graupel')
    return shape


def get_m(T, freq):
    """
    Complex refractive index for hydrometeors with given temperature and
    frequency.
    """
    rain = get_m_rain(T, freq)
    snow = get_m_snow(T, freq)  # placeholder!
    hail = get_m_hail(T, freq)  # placeholder!
    mix = get_m_mix(T, freq)  # placeholder!
    graupel = get_m_graupel(T, freq)  # placeholder!

    m = ph.Array.merged([rain, hail, snow, mix, graupel], xdim='htype')
    return m


def get_m_rain(T, freq):
    """
    Complex refractive index for raindrops with given temperature and
    frequency.

    According to empirical relation by Liebe et al. (1991)
    """
    theta = 1 - 300 / T
    etha_0 = 77.66 - 103.3 * theta
    etha_inf = 0.066 * etha_0
    gamma_D = (20.27 + 146.5 * theta + 314 * theta**2) * 1e9
    etha_D = (etha_0 - etha_inf) / (1 - 1j * freq / gamma_D) + etha_inf
    m = uf.sqrt(etha_D)

    m.setattrs(htype='rain')
    m.setname('refractive index')
    return m


# placeholders!
def get_m_snow(T, freq):
    m = T**0
    m.setattrs(htype='snow')
    m.setname('refractive index')
    return m


def get_m_hail(T, freq):
    m = T**0
    m.setattrs(htype='hail')
    m.setname('refractive index')
    return m


def get_m_mix(T, freq):
    m = T**0
    m.setattrs(htype='mix')
    m.setname('refractive index')
    return m


def get_m_graupel(T, freq):
    m = T**0
    m.setattrs(htype='graupel')
    m.setname('refractive index')
    return m


###############################################################################
# harray wrappers
dropshape = uf.ufunc(scat.shapes.dropshape)
dropshape2 = uf.ufunc(scat.shapes.dropshape2)
dropshape3 = uf.ufunc(scat.shapes.dropshape3)

def get_S(D, wl, mm, sfunc, theta0, theta, phi0, phi, alpha, beta, **kwargs):
    """
    Calculate the scattering amplitude matrix S.

    Returns:
        The amplitude (S) matrix.
    """
    def calc_S(D, wl, mm, sfunc, theta0, theta, phi0, phi, alpha, beta,
               **kwargs):
        T = scat.calc_T(D, wl, mm, sfunc=sfunc, **kwargs)
        S = np.flip(scat.calc_S(T, theta0, theta, phi0, phi, alpha, beta), axis=(-1,-2))
        return S


    dims = [ph.Index(['H', 'V'], 'out_pol'),
            ph.Index(['H', 'V'], 'in_pol')]
    S = uf.sufunc(dtypes={'float64':'complex128'})(calc_S)(D, wl, mm,
                 sfunc, theta0, theta, phi0, phi, alpha, beta, new_index=dims,
                 **kwargs)
    return S


def get_Z(S):
    """
    Calculate the scattering phase matrix Z.

    Returns:
        The phase (Z) matrix.
    """
    dims = [ph.Index(['I', 'Q', 'U', 'V'], 'in_stokes'),
            ph.Index(['I', 'Q', 'U', 'V'], 'out_stokes')]
    Z =uf.ufunc(scat.calc_Z)(S, new_index=dims, drop_dim=('out_pol',
                                                              'in_pol'))
    Z.setname('phase')
    return Z


def get_S_oa(D, wl, mm, sfunc, theta0, theta, phi0, phi, wfunc, n_alpha=5,
              **kwargs):
    """
    Compute the S matrix using variable orientation scatterers.
    average over orientation.

    Returns
    -------
    S : Array
        The amplitude matrix.
    """
    def calc_S_oa(D, wl, mm, sfunc, theta0, theta, phi0, phi, wfunc, n_alpha=5,
                  **kwargs):
        T = scat.calc_T(D, wl, mm, sfunc=sfunc, **kwargs)
        S = np.flip(scat.calc_S_oa(T, theta0, theta, phi0, phi, wfunc, n_alpha=5), axis=(-1, -2))
        return S


    dims = [ph.Index(['H', 'V'], 'out_pol'),
            ph.Index(['H', 'V'], 'in_pol')]
    S = uf.sufunc(dtypes={'float64':'complex128'})(calc_S_oa)(D, wl, mm, sfunc,
                 theta0, theta, phi0, phi, wfunc, n_alpha=5, new_index=dims,
                 **kwargs)
    return S


def get_SZ_oa(D, wl, mm, sfunc, theta0, theta, phi0, phi, wfunc, n_alpha=5,
              **kwargs):
    """
    Compute the S and Z matrices using variable orientation scatterers.
    average over orientation.

    Returns
    -------
    S : Array
        The amplitude matrix

    Z : Array
        The phase matrix.
    """
    def calc_SZ_oa(D, wl, mm, sfunc, theta0, theta, phi0, phi, wfunc,
                   n_alpha=5, **kwargs):
        T = scat.calc_T(D, wl, mm, sfunc=sfunc, **kwargs)
        S, Z = scat.calc_S_oa(T, theta0, theta, phi0, phi, wfunc, n_alpha=5)
        return S, Z


    S, Z = uf.sufunc(dtypes={'float64':'complex128'})(calc_SZ_oa)(D, wl, mm,
                     sfunc, theta0, theta, phi0, phi, wfunc, n_alpha=5,
                     **kwargs)
    return S, Z


###############################################################################
if __name__ == '__main__':
    """
    Example of usage.
    """
    from mwlink import inout as io

    attrs = {'quantity':'diameter', 'unit':'m'}
    D = ph.arange(1e-5, 7e-2, 1e-5, 'diameter', attrs=attrs)

    attrs = {'quantity':'temperature', 'unit':'K'}
    T = ph.arange(263, 313, 0.5, 'temperature', attrs=attrs)

    T = ph.arange(288, 289, 1, 'temperature', attrs=attrs)
#    D = ha.arange(1e-3, 2e-3, 1e-3, 'diameter', attrs=attrs)

    attrs = {'quantity':'frequency', 'unit':'Hz'}
#    for f in tqdm(range(5, 10, 5)):
#        freq = ph.arange(f*1e9, f*1e9 + 5e9, 1e9, 'frequency', attrs=attrs)
#        outdat = calc_scatter(freq, D, T)
#
#        outdat.attrs['level'] = 'scatter'
#        outdat.name = 'simulated'
#        outdat.attrs['pro_id'] = 'thurai_full_regfreq'
#        print('writing to database')
#        io.export_ds(outdat)

    freq = ph.index([73.5E9, 83.5E9], name='frequency', attrs=attrs)
    outdat = calc_scatter(freq, D, T)
    outdat.attrs['level'] = 'scatter'
    outdat.attrs['pro_id'] = 'dropshape_thurai07'
    outdat.name = 'simulated'
    print('writing to database')
    io.export_ds(outdat)


