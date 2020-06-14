# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:51:04 2015

@author: T.C. van Leth

TODO: Rayleigh scattering for NIR/optical links
TODO: calculate Johnson SB and lognormal distributions
"""

import logging

import numpy as np
import pyhad as ph
from pyhad import ufuncs as uf
from scipy.interpolate import interp1d


def disdro_l2(indat, scat=None):
    """
    calculate v(D) and N(D)
    """
    logging.info('calculating bulk quantities from disdrometers')

    count = indat.aselect(quantity='particle_count')
    v = indat.aselect(name='velocity')
    D = indat.aselect(name='diameter')
    dD = indat.aselect(name='diameter_binwidth')
    dv = indat.aselect(name='velocity_binwidth')
    hmatrix = indat.aselect(quantity='hydrometeor_composition')

    dt = indat.getstep('time')/np.timedelta64(1, 's')
    L = indat.getattrs('beam_length')
    W = indat.getattrs('beam_width')

    dat = []
    n_vD = get_nvD(count, D, dD, v, dv, dt, L, W)
    dat.append(n_vD)
    dat.append(hmatrix)
    if scat is not None:
        D = scat['diameter']
        dD = D.step['diameter']
        n_vD = n_vD.reindex(diameter=D, how='nearest') # 1
        hmatrix = hmatrix.reindex(diameter=D, how='nearest') # 1
        n_vD = n_vD * hmatrix.sel(htype='rain')
        del n_vD[:, :, 'diameter_binwidth']
        del hmatrix[:, :, 'diameter_binwidth']

#    v_D = get_vD(n_vD, v, dv)
    n_D = get_ND(n_vD, dv)
    dat.append(n_D)
    if scat is not None:
        dpc = scat.aselect(name='dpc').rechunk({'frequency': 1})
        ref = scat.aselect(name='refl').rechunk({'frequency': 1})
        ext = scat.aselect(name='ext').rechunk({'frequency': 1})
        phi = get_diffphase(n_D, dpc, dD)
        Z = get_reflec(n_D, ref, dD)
        k = get_atten(n_D, ext, dD)
        dat += [phi, Z, k]

    #  maybe use actual rho_w and rho_a?
    m_w = get_dropmass(D)
#    m_s = get_snowmass(D)
#    m_i = get_icemass(D)  # placeholder
#    m_g = get_graupelmass(D)
#    htype = ha.index(['rain', 'hail', 'graupel', 'snow'], 'htype')
#    mass = ha.merge([m_w, m_s, m_i, m_m, m_g], xdim=htype)

    R_new = get_flux(n_vD, m_w, v, dD, dv)
    lwc = get_LWC(n_D, m_w, dD)
    dat += [R_new, lwc]

#    stdv_D = get_stdD(n_D, D, dD)
#    expo_dsd = get_expoDSD(n_D, D, dD)
    gamm_dsd = get_gammaDSD(n_D, D, dD)
#    test_dsd = get_testudDSD(n_D, D, dD)
    dat += [*gamm_dsd]
    return ph.merge(dat) #1


###############################################################################
def get_nvD(count, D, dD, v, dv, dt, L, W):
    """
    calculate particle density from particle count
    """
    A = (-D * 0.5 + W) * L
    n_vD = count / (v * A * dD * dv * dt)

    n_vD.setattrs(quantity='particle_density', unit='m^{-5}\,s')
    n_vD.setattrs(name='particle_size_velocity_distribution')
    n_vD.setattrs(sampling='mean')
    n_vD.setname('N(v,D)')
    return n_vD


def get_ND(n_vD, dv):
    """
    calculate particle size distribution
    """
    n_D = uf.sum(n_vD * dv, dim='velocity')

    n_D.setattrs(quantity='particle_size_density', unit='m^{-4}')
    n_D.setattrs(name='particle_size_distribution')
    n_D.setname('N(D)')
    return n_D


def get_vD(n_vD, v, dv):
    """
    calculate the velocity as a function of particle diameter
    """
    v_D = uf.average(v, n_vD, dim='velocity')

    v_D.setattrs(quantity='velocity', unit='m\,s^{-1}')
    v_D.setattrs(name='particle_velocity_distribution')
    v_D.setname('v(D)')
    return v_D
###############################################################################


def get_snowmass(D):
    """
    masses of snowflakes with diameters D
    """
    #  dry aggregate snow (Matrosov, 2007)
    m1 = 3.0E-2 * D**2
    m2 = 6.6E-1 * D**2.5
    m3 = 4.7E-0 * D**3

    c1 = (D > 1.0E-4) & (D <= 2.0E-3)
    c2 = (D > 2.0E-3) & (D <= 2.0E-2)
    c3 = D > 2.0E-2
    m = m1 * c1 + m2 * c2 + m3 * c3

    m.setattrs(quantity='mass', unit='kg', name='particle_mass')
    m.setname('m')
    return m


def get_dropmass(D, rho_w=1000.):
    """
    masses of raindrops with volume-equivalent diameter D
    """
    m = rho_w * (np.pi/6) * D**3

    m.setattrs(quantity='mass', unit='kg', name='particle_mass')
    m.setname('m')
    return m


def get_icemass(D, frac=1.0, rho_i=934.):
    """
    masses of ice pellets/hail stones with volume-equivalent diameter D
    """
    # placeholder!
    m = frac * rho_i * (np.pi/6) * D**3

    m.setattrs(quantity='mass', unit='kg', name='particle_mass')
    m.setname('m')
    return m


def get_graupelmass(D):
    """
    mass of graupel particles (Heymsfield & Wright, 2014)
    """
    rho = 0.18 * D**0.33
    m = rho * (np.pi/6) * D**3

    m.setattrs(quantity='mass', unit='kg', name='particle_mass')
    m.setname('m')
    return m


###############################################################################
def get_LWC(N_D, mass, dD):
    """
    liquid water content of precipitation
    """
    lwc = (mass * N_D * dD).sum(dim='diameter')

    lwc.setattrs(quantity='mass', unit='kg', name='liquid_water_content')
    lwc.setname('lwc')
    return lwc


def get_stdD(N_D, D, dD):
    """
    standard deviation of particle diameter
    """
    M0 = (N_D * dD).sum(dim='diameter')
    M1 = (N_D * D * dD).sum(dim='diameter')
    M2 = (N_D * D**2 * dD).sum(dim='diameter')

    M2 = uf.nandiv(M2, M0)
    M1 = uf.nandiv(M1, M0)**2
    sigmaD = uf.nansqrt(M2 - M1)

    sigmaD.setattrs(quantity='diameter', unit='m', sampling='stdv')
    sigmaD.setattrs(name='particle_diameter_spread')
    sigmaD.setname('stdv_D')
    return sigmaD


def get_flux(n_vD, mass, v, dD, dv):
    """
    (liquid water equivalent) precipitation intensity from dsd
    """
    R = uf.sum(mass * n_vD * v * dD * dv, dim=['diameter', 'velocity'])
    R *= 1E-3  # convert to volume flux (liquid water)
    R *= 3.6E6  # convert to mm/hr

    R.setattrs(quantity='volume_flux', unit='mm/hr')
    R.setattrs(name='precipitation_intensity')
    R.setname('precip')
    return R


###############################################################################
def get_testudDSD(N_D, D, dD):
    """
    Testud dsd parameterization (Testud, 2001)
    """
    M3 = (N_D * D**3 * dD).sum(dim='diameter')
    M4 = (N_D * D**4 * dD).sum(dim='diameter')

    N_0 = (4**4. / 6.) * uf.nandiv(M3**5, M4**4)

    N_0.setname('test_dsd_N_0')
    return N_0


def get_expoDSD(N_D, D, dD):
    """
    exponential dsd parameterization (Waldvogel, 1973)
    N(D) = N_0*exp(-Lamb*D)
    """
    M3 = (N_D * D**3 * dD).sum(dim='diameter')
    M6 = (N_D * D**6 * dD).sum(dim='diameter')

    lam = 120**(1/3) * uf.nandiv(M3, M6)**(1/3)
    N_0 = (1/6) * lam**4 * M3

    N_0.setname('exp_dsd_N_0')
    lam.setname('exp_dsd_lambda')
    return N_0, lam


def get_gammaDSD(N_D, D, dD):
    """
    gamma dsd parameterisation (Tokay & Short, 1996)
    N(D) = N_0*D**mu*exp(-Lamb*D)
    """
    M3 = (N_D * D**3 * dD).sum(dim='diameter')
    M4 = (N_D * D**4 * dD).sum(dim='diameter')
    M6 = (N_D * D**6 * dD).sum(dim='diameter')

    G = uf.nandiv(M4**3, M6 * M3**2)
    mu = uf.nandiv(5.5 * G - 4 + uf.sqrt(G * (G * 0.25 + 2)), 1 - G)
    lam = (mu + 4) * uf.nandiv(M3, M4)
    with np.errstate(over='ignore', invalid='ignore'):
        N_0 = uf.nandiv(lam**(mu+4), uf.gamma(mu+4))*M3

    N_0.setname('gamm_dsd_N_0')
    lam.setname('gamm_dsd_lambda')
    mu.setname('gamm_dsd_mu')
    return N_0, lam, mu


def get_johnsonDSD(N_D):
    """
    Johnson SB dsd parameterization (Johnson, 1949)
    """
    return None  # placeholder!


def get_lognDSD(N_D):
    """
    lognormal dsd parameterization
    """
    return None  # placeholder!


###############################################################################
def get_reflec(N_D, cross, dD):
    """
    radar reflectivity
    """
    Z = uf.nansum(N_D * cross, dim='diameter') * dD
    Z = 10 * uf.nanlog10(Z * 1e18)  # convert to dBZ

    Z.setattrs(quantity='radar_reflectivity', unit='dBZ')
    Z.setname('Z')
    return Z


def get_atten(N_D, cross, dD):
    """
    specific attenuation
    """
    k = uf.nansum(N_D * cross, dim='diameter') * dD
    k *= 1e4 / uf.log(10)  # convert to dB/km

    k.setattrs(quantity='specific_attenuation', unit='dB/km')
    k.setname('k')
    return k


def get_diffphase(N_D, cross, dD):
    """
    differential phase angle

    Parameters
    ----------
    N_D : data_container
        drop size distrubution

    returns
    -------
    """
    phi = uf.nansum(N_D * cross, dim='diameter') * dD
    phi *= 1000  # convert to rad/km

    phi.setattrs(quantity='differential_phase_angle', unit='rad/km')
    phi.setname('phi')
    return phi


###############################################################################
def resample_dsd(ND, scat):
    ND = ND / 1000
    time = ND['time']
    D_pars = ND['diameter'] * 1000
    D = scat['diameter'] * 1000
    D1 = D.values
    D1 = D1[(D1>=D_pars.values[0]) & (D1<=D_pars.values[-1])]
    D.values = D1
    dD = D1[1]-D1[0]

    # interpolation of dsd
    func = interp1d(D_pars, ND, kind='linear', bounds_error=False, axis=0)
    NDpath = ph.Array(func(D1), coords=[D, time])
    NDpath.setname('original_ND')
    return NDpath, dD