#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:03:48 2018

@author: tcvanleth
"""
import phad as ph
from phad import ufuncs as uf


def get_vt_rain(rho_w, rho_a, eta, sigma, D):
    """
    terminal velocity for rain (Beard 1976)
    NB: not Beard (1977)!
    """
    d_rho = rho_w - rho_a  # density difference between drop and air

    # for D < 1.07 mm
    C_sc = 1.0  # slip correction factor
    N_Da = 4 * rho_a * d_rho * ph.constants.g / (3*eta**2) * D**3
    X = uf.log(N_Da)
    Y = (-3.18657E+0 +
         9.92696E-1 * X +
         -1.53193E-3 * X**2 +
         -9.87059E-4 * X**3 +
         -5.78878E-4 * X**4 +
         8.55176E-5 * X**5 +
         -3.27815E-6 * X**6)
    Re_1 = C_sc * uf.exp(Y)

    # for 1.07 mm < D < 7 mm
    B_0 = 4 * d_rho * ph.constants.g / (3 * sigma) * D**2
    N_P = sigma**3 * rho_a**2 / (eta**4 * d_rho * ph.constants.g)
    X = uf.log(B_0 * N_P**(1/6.))
    Y = (-5.00015E+0 +
         5.23778E+0 * X +
         -2.04914E+0 * X**2 +
         4.75294E-1 * X**3 +
         -5.42819E-2 * X**4 +
         2.38449E-3 * X**5)
    Re_2 = N_P**(1/6.) * uf.exp(Y)

    c = D < 1.07E-3
    Re = c * Re_1 + ~c * Re_2
    vt = eta * Re / (rho_a * D)

    vt.setattrs(quantity='velocity', unit='m/s')
    vt.setattrs(name='terminal_velocity')
    vt.delattrs('material', 'ventilated', 'wet')
    vt.setname('v_term')
    return vt


def get_vt_snow(T, P, D):
    """
    terminal velocity for snow (Brandes et al., 2008)
    """
    T_0 = 273.14  # 0 deg celsius
    vt = 0.884 * 0.035 * (T - T_0) * (D * 1000)**0.237 * (1E5 / P)**0.545

    vt.setattrs(quantity='velocity', unit='m/s')
    vt.setattrs(name='terminal_velocity')
    vt.delattrs('ventilated', 'wet')
    vt.setname('v_term')
    return vt


def get_vt_hail2(P, D):
    """
    terminal velocity for hail (Heymsfield & Wright, 2014)
    """
    v1 = 12.65 * (D/0.01)**0.65 * (1E5/P)**0.545
    v2 = 15.65 * (D/0.01)**0.35 * (1E5/P)**0.545
    c = D > 20.5e-3
    vt = c * v1 + ~c * v2

    vt.setattrs(quantity='velocity', unit='m/s')
    vt.setattrs(name='terminal_velocity')
    vt.setname('v_term')
    return vt


def get_vt_hail(rho_a, rho_b, eta_a, D):
    """
    terminal velocity for hail, sleet, graupel (Heymsfield & Wright, 2014)
    """
    X = (4/3) * D**3 * rho_b * ph.constants.g * rho_a / eta_a**2

    cond = X < 6.77e4
    Re = cond * 0.106 * X**0.693 + ~cond * 0.55 * X**0.545

    vt = Re * eta_a / (rho_a * D)

    vt.setattrs(quantity='velocity', unit='m/s')
    vt.setattrs(name='terminal_velocity')
    vt.setname('v_term')
    return vt


def get_vt_graupel(rho_a, eta_a, D):
    """
    terminal velocity for graupel (Heymsfield et al., 2014)
    """
    rho_b = 1800 * D**0.33
    vt = get_vt_hail(rho_a, rho_b, eta_a, D)
    return vt