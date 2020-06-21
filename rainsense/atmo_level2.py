# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:35:04 2015

@author: T.C. van Leth

TODO: include lapse rate
"""
import logging
import pyhad as ph


def atmo_l2(indat):
    """
    calculate theoretical terminal velocities for parsivel diameter classes
    """
    logging.info('calculating secondary atmospheric characteristics')
    P = indat.aselect(quantity='pressure')
    T = indat.aselect(quantity='temperature', ventilated=True, wet=False)

    rho_a = get_rho_air(T, P)
    rho_w = get_rho_water(T)
    eta_a = get_eta_air(T)
    sigma = get_sigma(T)
    return ph.merge([rho_a, rho_w, eta_a, sigma, P, T])


###############################################################################
def get_rho_air(T, P):
    """
    density of air based on temperature and pressure
    """
    rho = 3.48E-3 * P / T

    rho.setattrs(quantity='density', unit='kg/m3')
    rho.setattrs(material='air')
    rho.setattrs(name='air_density')
    rho.setname('rho_a')
    return rho


def get_rho_water(T):
    """
    density of liquid water based on ambient temperature
    """
    T_Cel = T - 273.14  # unit conversion

    # density of liquid water above 0 according to Kell (1972)
    A = [-3.932952E-10,
         +1.497562E-7,
         -5.544846E-5,
         -7.922210E-3,
         +1.8224944E+1,
         +9.998396E+2]
    B = 18.159725E-3
    Y = (A[0] * T_Cel**5 +
         A[1] * T_Cel**4 +
         A[2] * T_Cel**3 +
         A[3] * T_Cel**2 +
         A[4] * T_Cel +
         A[5])
    rho1 = Y / (1 + B * T_Cel)

    # density of liquid water below 0 according to Boyd (1951)
    A = [-1.08E-2,
         8.60E-2,
         9.9984E+2]
    rho2 = A[0] * T_Cel**2 + A[1] * T_Cel + A[2]
    c = T_Cel >= 0
    rho = c * rho1 + ~c * rho2

    rho.setattrs(quantity='density', unit='kg/m3')
    rho.setattrs(material='liquid_water')
    rho.setattrs(name='liquid_water_density')
    rho.setname('rho_w')
    return rho


def get_eta_air(T):
    """
    dynamic viscosity of air
    """
    # from Beard (1977)
    eta = 1.832E-5 * (1 + 0.00266 * (T - 296))

    eta.setattrs(quantity='dynamic_viscosity', unit='kg/(m*s)')
    eta.setattrs(material='air')
    eta.setattrs(name='air_dynamic_viscosity')
    eta.setname('eta_a')
    return eta


def get_sigma(T):
    """
    surface tension of water in air (Vargaftik, Volkov & Voljak, 1983)
    """
    B = 235.8E-3  # N/m
    b = -0.625
    mu = 1.256
    T_c = 647.15  # critical temperatur in Kelvin
    sigma = B * ((T_c - T) / T_c)**mu * (1 + b * (T_c - T) / T_c)

    sigma.setattrs(quantity='surface_tension', unit='N/m')
    sigma.setattrs(material='air-water')
    sigma.setattrs(name='surface_tension_of_water_in_air')
    sigma.setname('sigma_wa')
    return sigma


if __name__ == '__main__':
    from mwlink import inout as io
    ph.common.standardlogger()

    setID = 'Veenkampen'
    proID = 'atmo_test1'

    data = io.import_ds('atmo_l1', setID, conform=True)
    l2dat = atmo_l2(data)
    io.export_ds(l2dat, level='atmo_l2', pro_id=proID)
