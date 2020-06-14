# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:38:06 2015

T.C. van Leth
"""

import logging

import numpy as np
import pyhad as ph
from pyhad import ufuncs as uf


def link_l1b(indat, calib):
    """
    Convert raw detector voltages to power.
    """
    logging.info('converting detector voltages to power(dB)')
    volt = indat.select(quantity='voltage', side='receiver')
    calib = calib.aselect(quantity='voltage', side='receiver')

    PdB = get_power(volt, calib)

    PdB.setattrs(quantity='power', unit='dBm')
    return PdB


def link_l1c(indat, aux):
    """
    Convert all link received and transmitted powers to attenuation and dpa.
    """
    logging.info('calculating link attenuation')
    L = linklength(indat)
    Rx = indat[..., ..., 'rx']
    Q = Rx[..., 'channel_3']
    I = Rx[..., 'channel_4']

    HV = Rx.select(polarization=['V', 'H'])
    Tx = get_tx(indat)
    wet = aux.aselect(quantity='hydrometeor_type') == 'rain'

    Rref = get_ref(HV, wet, how='Overeem11')
    att = get_att(HV, L, Tx, Rref, wet)
    phi = get_dpa(Q, I, L)
    out = ph.merge([Rref, att, phi])
    return out


def link_l2(indat, powlaws):
    """
    Convert all channel attenuations and dpa to rain intensities.
    """
    logging.info('calculating link rain rates')
    k = indat[..., 'k']

    pars = powlaws.select(quantity='specific_attenuation',
                          frequency=k.getattrs('frequency'),
                          polarization=k.getattrs('polarization'))

    print(k)
    print(pars)
    R = get_rain(k, pars)

    R.setattrs(quantity='rain_intensity', unit='mm/hr')
    R.setname('precip', include=['sampling'])
    return R


def linklength(indat):
    """
    Determine link length in meter from the endpoint spatial coordinates.
    """
    y1 = indat.getattrs('site_a_latitude')
    x1 = indat.getattrs('site_a_longitude')
    z1 = indat.getattrs('site_a_altitude')
    y2 = indat.getattrs('site_b_latitude')
    x2 = indat.getattrs('site_b_longitude')
    z2 = indat.getattrs('site_b_altitude')
    return ph.geometry.length(x1, x2, y1, y2, z1, z2)


###############################################################################
@uf.ufunc
def get_power(volt, pars):
    '''
    Convert detector voltage to power.
    '''
    if pars[0] == 'log':
        PdB = pars[1] * volt + pars[2]
    if pars[0] == 'lin':
        PdB = 10 * uf.log10(pars[1] * volt + pars[2])
    return PdB


def get_tx(indat):
    """
    Match transmitted power with retrieved power based on sampling strategy.
    """
    if indat.getattrs('atpc') is 'on':
        Tx = indat.select(quantity='power', side='transmitter')
        isample = indat.getattrs('sampling')
        if isample == 'max':
            Tx = Tx.aselect(sampling='min')
        elif isample == 'min':
            Tx = Tx.aselect(sampling='max')
        else:
            Tx = Tx.aselect(sampling=isample)
    else:
        try:
            Tx = indat.getattrs('const_tx')
        except KeyError:
            Tx = 0
    return Tx


def get_att(Rx, L, Tx, Rref, wet):
    """
    Convert Rx and Tx to specific attenuation.
    """
    Rx -= Tx
    k = uf.maximum(Rref - Rx, 0) * wet / (1000 / L)

    k.setattrs(quantity='specific_attenuation', unit='dB/km')
    k.setname('k')
    return k


def get_dpa(Q, I, L):
    """
    Convert quadrature and in_phase signals to differential phase angle.
    """
    phi = -uf.arctan2(Q, I) * (1000 / L)

    phi.setattrs(quantity='differential_phase_angle', unit='rad/km')
    phi.setname('phi')
    return phi


def get_ref(Rx, wet=None, how=None, period=24):
    '''
    Determine baseline in receiver signal.
    '''
    if how is None:
        how = 'max'

    if how == 'max':
        ref = uf.nanrollmax(Rx, window={'time':np.timedelta64(period, 'h')})
    elif how == 'Overeem11':
        # median of all dry values in surrounding period
        if wet is None:
            Exception
        ref = Rx[wet == False]
        ref = uf.nanrollmedian(ref, window={'time':np.timedelta64(period, 'h')})
    elif how == 'Chwala12':
        pass
    elif how == 'Wang12':
        pass
    else:
        Exception

    ref.setattrs(quantity='power', unit='dB', baseline_determination_method=how)
    ref.setname('rxref')
    return ref


def get_wetdry(Rx, L, x0, x1, y0, y1, radius=15000):
    """
    Determine wet and dry periods using neighbouring links.
    """
    wet = []
    for link_id in Rx['link_name']:
        # calculate distance between links
        l0 = ph.geometry.length(x0[link_id], y0[link_id], x0, y0)
        l1 = ph.geometry.length(x1[link_id], y1[link_id], x1, y1)
        distance = uf.minimum(l0, l1)

        # determine neighbouring links
        if uf.nansum(distance < radius, dim='link_name') < 3:
            continue
        L = L[distance < radius]
        Rx = Rx[distance < radius]

        # Calculate attenuation wrt 24 hour maximum
        att = Rx - uf.nanrollmax(Rx, window={'time':np.timedelta64(24, 'h')})
        spec_att = att / L

        # classify wet periods based on spatial median attenuation
        iwet = ((uf.median(att, dim='link_name') < -1.4) &
                (uf.median(spec_att, dim='link_name') < -0.7))
        iwet |= uf.rollmin(att, window={'time':np.timedelta64(1, 'h')}) < -2.
        wet += [iwet]

    wet = ph.merge(wet, xdim=Rx.indexes['link_name'])
    wet.setname('wet')
    wet.setattrs(quantity='wetdry_classification', unit='true/false')
    return wet


def wet_corr(A, how=None):
    """
    Correct measured attenuation for attenuation due to wet antennas.
    """
    if how is None:
        how = 'Leijnse07'

    if how == 'Leijnse07':
        C1 = 3.32
        C2 = 0.48
        A_w = uf.maximum(uf.minimum(C1 * (1 - uf.exp(-C2 * A)), A), 0)
        A = A - A_w
    elif how == 'KhaRo01':
        C1 = 8.0
        C2 = 0.125
        A_w = uf.maximum(uf.minimum(C1 * (1 - uf.exp(-C2 * A)), A), 0)
    elif how == 'MinNak':
        pass
    elif how == 'Overeem11':
        A = A - 2.3
    else:
        Exception
    return A


def temp_corr(A, T, how=None):
    """
    Correct measured attenuation for temperature dependence in analog electronics.
    """
    return A


@uf.ufunc
def get_rain(k, pars):
    """
    Convert attenuation to rain intensity.
    """
    return pars[0] * k**pars[1]
