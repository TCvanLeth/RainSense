# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:44:11 2015

@author: T.C. van Leth

NB: solid and mixed precipitation is left unadjusted!
The code incorporates velocity adjustment for solid but not for mixed.
TODO: determine aggregation for velocity correction
"""
import os, logging

import numpy as np
import phad as ph
from phad import ufuncs as uf

from mwlink.process import particle_velocities as pv


def pars_correction(indat, atmo):
    """
    rescale velocity data and mask outliers(step 1 of Raupach & Berne, 2015)
    apply correction factor to DSD (step 2 of Raupach & Berne 2015)
    """
    logging.info('applying dsd correction')

    count = indat.aselect(quantity='particle_count')
    R_old = indat.aselect(quantity='rain_intensity')
    v = indat.aselect(name='velocity')
    dv = indat.aselect(name='velocity_binwidth')
    D = indat.aselect(name='diameter')

    rho_w = atmo[..., 'rho_w']
    rho_a = atmo[..., 'rho_a']
    eta_a = atmo[..., 'eta_a']
    sigma  = atmo[..., 'sigma_wa']
    T = atmo[..., 'temp_vent_dry']
    P = atmo[..., 'pressure']

    vt_rain = pv.get_vt_rain(rho_w, rho_a, eta_a, sigma, D)
    hmatrix = mask_types(D, T, P, rho_w, rho_a, eta_a, sigma, v)
    rain = hmatrix.sel(htype='rain')
    count = rescale_vel(count, vt_rain, rain, v, dv)
    count *= get_corf(R_old, rain)
    new = ph.merge([count, hmatrix]) # 1
    return new


def rescale_vel(count, v_t, rain, v, dv):
    """
    calculate the differences beteen predicted and measured velocities and
    put counts in new bins accordingly
    """
    # calculate differences between measured velocity and theoretical velocity
    v_diff = v - v_t

    # calculate mean bias per instrument and diameter class
    bias = uf.nanaverage(v_diff, count * rain, dim=['velocity', 'time'])

    # calculate  bias corrected velocities
    v_new = v - bias

    # rebin velocity
    v_edge = v - 0.5 * dv
    index = uf.searchsorted(v_edge, v_new) - 1
    newcount = uf.bincount(index, count, dim='velocity')
    newcount.setattrs(quantity='particle_count', unit='particles',
                      name='hydrometeor_count')
    newcount.setname('h_count')
    return newcount


def mask_types(D, T, P, rho_w, rho_a, eta_a, sigma, v):
    """
    create unique mask for each record.
    rain mask based on Raupach & Berne (2015)
    hail and graupel velocity curves from Heymsfield & Wright (2014)
    hail and graupel masks based on Friedrich et al. (2013)
    snow velocity curve from Brandes et al. (2008)
    """
    vt_rain = pv.get_vt_rain(rho_w, rho_a, eta_a, sigma, D)
    vt_hail_min = pv.get_vt_hail(rho_a, 200, eta_a, D)
    vt_hail_max = pv.get_vt_hail(rho_a, 800, eta_a, D)
    vt_grau_min = pv.get_vt_hail(rho_a, 100, eta_a, D)
    vt_grau_max = pv.get_vt_graupel(rho_a, eta_a, D)
    vt_snow = pv.get_vt_snow(T, P, D)

    rain = ((D < 7.5e-3) & (v >= 0.5) &
            (((v >= vt_rain - 3) & (D < 5e-3)) |
             ((v >= (vt_rain + vt_hail_max) * 0.5) & (D >= 5e-3))) &
            (v < vt_rain + 4))

    hail = ((D >= 5e-3) & (v >= vt_hail_min * 0.6) &
            (((v < vt_hail_max * 1.4) & (D >= 7.5e-3)) |
             ((v < (vt_hail_max + vt_rain) * 0.5) & (D < 7.5e-3))))

    grau = ((D >= 2e-3) & (D < 5e-3) &
            (v >= (vt_grau_min + vt_snow) * 0.5) &
            (v < vt_grau_max))

    snow = ((v >= vt_snow * 0.6) &
            (((v < vt_snow * 1.4) & (D >= 5e-3)) |
             ((v < (vt_snow + vt_grau_min) * 0.5) & (D >= 2e-3) & (D < 5e-3))) &
            (T < 276.))

    htype = ph.Index(['rain', 'hail', 'graupel', 'snow'], 'htype')
    hmatrix = ph.merge([rain, hail, grau, snow], xdim=htype)
    hmatrix.setattrs(unit='particle_class_presence',
                     quantity="hydrometeor_composition", sampling='mean',
                     name='hydrometeor_composition')
    hmatrix.setname('h_comp')
    hmatrix = hmatrix.astype(int)
    return hmatrix


def get_corf(R_old, rain):
    # get the correction factors from file
    inpath = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0],
                          'Parameters',
                          'N_D_corr_fact_rain.dsd')
    D = []
    c_facts = []
    with open(inpath, 'r') as inh:
        i = 0
        for line in inh:
            line = line.strip('\n\r')
            row = line.split()

            # get the rain intensity class boundaries
            if i == 1:
                R_bin = np.asarray(row[1:], dtype=float)

            # get the correction factors for each rain intensity class
            # and each diameter class
            if i > 1:
                if len(row) != 0:
                    D.append(row[0])
                    c_facts.append(row[1:])
            i += 1
    attrs = {'quantity': 'diameter', 'unit': 'm'}
    D = ph.index(np.asarray(D).astype(float), 'diameter', attrs=attrs)
    corf = []
    for i in range(len(R_bin) - 1):
        corf += [ph.Array(np.asarray(c_facts, dtype=float)[:, i], coords=[D],
                          name='corf')]

    # obtain correction factors for each timestep
    c_fact = 0.
    for i in range(len(R_bin)-1):
        cond = (R_old >= R_bin[i]) & (R_old < R_bin[i+1]) #& rain
        c_fact = cond * corf[i] + c_fact
    #cond = ~rain
    #c_fact = cond * 1. + c_fact
    return c_fact


###############################################################################
# deprecation candidates

def precip_conv(indat):
    hcomp = indat.aselect(quantity='hydrometeor_composition')
    htypes = ph.index(['rain', 'snow', 'hail', 'mix', 'graupel'], 'htype')
    hcomp = (cons_htypes(hcomp) == htypes).astype(int)

    hcomp.setattrs(unit='particle_class_presence', sampling='mean')
    return indat.update(hcomp)


@uf.ufunc
def cons_htypes(synop):
    """
    make ndarray of simplified types
    """
    htype = np.zeros_like(synop, dtype='<U7')
    htype[np.in1d(synop, ['51', '53', '55', '58', '59', '61', '63', '65'])] = 'rain'
    htype[np.in1d(synop, ['89', '90'])] = 'hail'
    htype[np.in1d(synop, ['71', '73', '75', '77'])] = 'snow'
    htype[np.in1d(synop, ['68', '69', '87', '88'])] = 'mix'
    htype[synop == '00'] = 'dry'
    return htype
