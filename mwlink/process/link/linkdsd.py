#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:44:52 2018

@author: tcvanleth
"""

import os

from numba import jit
import numpy as np
from pyhad import ufuncs as uf
from scipy import optimize

from mwlink.process.link import solver


"""
            if (ier == 1 and sol[1] < 50 and sol[1] > 1 and sol[0] != x0 and
                sol[1] != x1 and sol[0] < 0.55*sol[1]+5 and sol[0] > 0.55*sol[1]-3
                and sol[0] < 1.9 * sol[1] - 2):
"""
#            out = optimize.fsolve(func2, [x0, x1], fprime=jac2, maxfev=maxfev,
#                                       full_output=True)
#            sol, info, ier, mesg = out

odir = "/home/tcvanleth/Data2/wurex_parsivel/dsd2"
grid = np.load(os.path.join(odir, 'mask.npy'))

@jit(nopython=True)
def checkmask(sol):
    dmu = 0.06
    dlam = 0.106
    return np.any((np.abs(sol[0] - grid[0]) < dmu) & (np.abs(sol[1] - grid[1]) < dlam))


@jit(nopython=True, cache=True)
def func2(x, args):
    D = args[0]
    k1 = args[1]
    k2 = args[2]
    k3 = args[3]
    sigma1 = args[4]
    sigma2 = args[5]
    sigma3 = args[6]

    N_D = D**x[0] * np.exp(-x[1]*D)
    return np.array([(k3/k1) * np.nansum(N_D * sigma1) - np.nansum(N_D * sigma3),
            (k1/k2) * np.nansum(N_D * sigma2) - np.nansum(N_D * sigma1)])


@jit(nopython=True, cache=True)
def jac2(x, args):
    D = args[0]
    k1 = args[1]
    k2 = args[2]
    k3 = args[3]
    sigma1 = args[4]
    sigma2 = args[5]
    sigma3 = args[6]

    N_D1 = D**x[0] * np.log(D) * np.exp(-x[1]*D)
    N_D2 = -D * D**x[0] * np.exp(-x[1]*D)
    return np.array([[(k3/k1) * np.nansum(N_D1 * sigma1) - np.nansum(N_D1 * sigma3),
                      (k3/k1) * np.nansum(N_D2 * sigma1) - np.nansum(N_D2 * sigma3)],
                     [(k1/k2) * np.nansum(N_D1 * sigma2) - np.nansum(N_D1 * sigma1),
                      (k1/k2) * np.nansum(N_D2 * sigma2) - np.nansum(N_D2 * sigma1)]])


# inverse modeling
#@jit(nopython=True, parallel=True)
def solve(k1, k2, k3, sigma1, sigma2, sigma3, D, dD):
    """
    find three parameter Gamma distribution function given three different
    microwave attenuations.
    """
    args = (D, k1, k2, k3, sigma1, sigma2, sigma3)

    if (k1 <= 1e-7 or k2 <= 1e-7 or k3 <= 1e-7 or np.isnan(k1) or np.isnan(k2)
        or np.isnan(k3)):
        return 0., np.nan, np.nan

    x0_max = 35
    x1_max = 70
    x0s = np.arange(-1, x0_max, 0.2)
    x1s = np.arange(0, x1_max, 0.2)
    for x0 in x0s:
        for x1 in x1s:
            maxfev = 200
            sol, ier = solver.hybrj(func2, jac2, args, np.array([x0, x1]), np.array([-1, 0]),
                                    np.array([x0_max, x1_max]), maxfev=maxfev)
            if (ier == 1 and sol[0] != x0 and sol[1] != x1 and checkmask(sol)):
                N0 = k1 / (dD * np.nansum(D**sol[0] * np.exp(-sol[1]*D) * sigma1))
                return N0, sol[0], sol[1]
    else:
        return np.nan, np.nan, np.nan


def solve2(k1, k2, sigma1, sigma2, D, dD):
    """
    find three parameter Gamma distribution function given two different
    microwave attenuations.
    """
    def func1a(x):
        lam = 2.5e-2*x**2 + 1.03*x + 1.97
        N_D = D**x * np.exp(-lam*D)
        return (k1/k2) * np.nansum(N_D * sigma2) - np.nansum(N_D * sigma1)

    # break off early when there is no rain or no data
    if k1 <= 1e-7 or k2 <= 1e-7 or np.isnan(k1) or np.isnan(k2):
        return 0., np.nan, np.nan

    # find bounds for root
    mus = np.arange(-1, 35.2, 0.2)

    ys = np.vectorize(func1a)(mus)
    diffsign = np.diff(np.sign(ys))
    idx = np.where((diffsign == 2) | (diffsign == -2))[0]
    lowers = mus[idx]
    uppers = mus[idx + 1]
    if len(lowers) > 0:
        if len(lowers) == 1:
            lower = lowers[0]
            upper = uppers[0]
        elif any(lowers > 0):
            lower = np.min(lowers[lowers > 0])
            upper = np.min(uppers[lowers > 0])
        else:
            lower = np.max(lowers)
            upper = np.max(uppers)

        # solve 1-parameter problem
        mu = optimize.brentq(func1a, lower, upper)

        # compute 2nd parameter
        lam = 2.5e-2 * mu**2 + 1.03 * mu + 1.97
        # compute 3rd parameter
        N0 = k1 / (dD * np.nansum(D**mu * np.exp(-lam * D) * sigma1))
        return N0, mu, lam
    else:
        return np.nan, np.nan, np.nan

def solve3(k1, k2, sigma1, sigma2, D, dD):
    """
    find three parameter Gamma distribution function given two different
    microwave attenuations.
    """
    def func1a(x):
        mu = 2
        N_D = D**mu * np.exp(-x*D)
        return (k1/k2) * np.nansum(N_D * sigma2) - np.nansum(N_D * sigma1)

    # break off early when there is no rain or no data
    if k1 <= 1e-7 or k2 <= 1e-7 or np.isnan(k1) or np.isnan(k2):
        return 0., np.nan, np.nan

    # find bounds for root
    mus = np.arange(-1, 35.2, 0.2)

    ys = np.vectorize(func1a)(mus)
    diffsign = np.diff(np.sign(ys))
    idx = np.where((diffsign == 2) | (diffsign == -2))[0]
    lowers = mus[idx]
    uppers = mus[idx + 1]
    if len(lowers) > 0:
        if len(lowers) == 1:
            lower = lowers[0]
            upper = uppers[0]
        elif any(lowers > 0):
            lower = np.min(lowers[lowers > 0])
            upper = np.min(uppers[lowers > 0])
        else:
            lower = np.max(lowers)
            upper = np.max(uppers)

        # solve 1-parameter problem
        lam = optimize.brentq(func1a, lower, upper)
        mu = 2
        # compute 3rd parameter
        N0 = k1 / (dD * np.nansum(D**mu * np.exp(-lam * D) * sigma1))
        return N0, mu, lam
    else:
        return np.nan, np.nan, np.nan



#def solve2(k1, k2, k3, sigma1, sigma2, sigma3, D, dD):
#    """
#    find three parameter Gamma distribution function given three different
#    microwave attenuations.
#    """
#    def func1a(x):
#        lam = 1.8e-6*x**4 - 6.3e-4*x**3 + 6.0e-2*x**2 + 4.8e-1*x + 2.53
#        N_D = D**x * np.exp(-lam*D)
#        return (k1/k2) * np.nansum(N_D * sigma2) - np.nansum(N_D * sigma1)
#
#    def func1b(x):
#        lam = 1.8e-6*x**4 - 6.3e-4*x**3 + 6.0e-2*x**2 + 4.8e-1*x + 2.53
#        N_D = D**x * np.exp(-lam*D)
#        return (k1/k3) * np.nansum(N_D * sigma3) - np.nansum(N_D * sigma1)
#
#    def func1c(x):
#        lam = 1.8e-6*x**4 - 6.3e-4*x**3 + 6.0e-2*x**2 + 4.8e-1*x + 2.53
#        N_D = D**x * np.exp(-lam*D)
#        return (k3/k2) * np.nansum(N_D * sigma2) - np.nansum(N_D * sigma3)
#
#    @jit(nopython=True)
#    def func2(x):
#        N_D = D**x[0] * np.exp(-x[1]*D)
#        return np.array([(k3/k1) * np.nansum(N_D * sigma1) - np.nansum(N_D * sigma3),
#                (k1/k2) * np.nansum(N_D * sigma2) - np.nansum(N_D * sigma1)])
#
#    @jit(nopython=True)
#    def jac2(x):
#      N_D1 = D**x[0] * np.log(D) * np.exp(-x[1]*D)
#      N_D2 = -D * D**x[0] * np.exp(-x[1]*D)
#      return np.array([[(k3/k1) * np.nansum(N_D1 * sigma1) - np.nansum(N_D1 * sigma3),
#                        (k3/k1) * np.nansum(N_D2 * sigma1) - np.nansum(N_D2 * sigma3)],
#                       [(k1/k2) * np.nansum(N_D1 * sigma2) - np.nansum(N_D1 * sigma1),
#                        (k1/k2) * np.nansum(N_D2 * sigma2) - np.nansum(N_D2 * sigma1)]])
#
#    # break off early when there is no rain or no data
#    if k1 <= 1e-7 and k2 <= 1e-7 and k3 <= 1e-7:
#        return 0., 0., 1.
#    if np.isnan(k1) or np.isnan(k2) or np.isnan(k3):
#        return np.nan, np.nan, np.nan
#
#    # find bounds for root
#    x0s = np.arange(-5, 100.2, 0.2)
#    for func in (func1a, func1b, func1c):
#        ys = np.vectorize(func)(x0s)
#        diffsign = np.diff(np.sign(ys))
#        idx = np.where((diffsign == 2) | (diffsign == -2))[0]
#        lowers = x0s[idx]
#        uppers = x0s[idx + 1]
#        if len(lowers) > 0:
#            if len(lowers) == 1:
#                lower = lowers[0]
#                upper = uppers[0]
#            elif any(lowers > 0):
#                lower = np.min(lowers[lowers > 0])
#                upper = np.min(uppers[lowers > 0])
#            else:
#                lower = np.max(lowers)
#                upper = np.max(uppers)
#
#            # solve 1-parameter problem
#            x0 = optimize.brentq(func, lower, upper)
#            x1 = 1.8e-6*x0**4 - 6.3e-4*x0**3 + 6.0e-2*x0**2 + 4.8e-1*x0 + 2.53
#
#            # solve 2-parameter problem
#            y0s = np.arange(-2, 100, 0.2)
#            y1pos = np.linspace(x1, x1 + 29.8, num=150)
#            y1neg = np.linspace(x1-0.2, x1 - 30, num=149)
#            y1s = np.empty((len(y1pos) + len(y1neg),))
#            y1s[::2] = y1pos
#            y1s[1::2] = y1neg
#            y0pos = np.linspace(x0, x0 + 29.8, num=150)
#            y0neg = np.linspace(x0-0.2, x0 - 30, num=149)
#            y0s = np.empty((len(y0pos) + len(y0neg),))
#            y0s[::2] = y0pos
#            y0s[1::2] = y0neg
#            for y0 in y0s:
#                if y0 < -5:
#                    continue
#                for y1 in y1s:
#                    if y1 < -1:
#                        continue
#                    sol, info = solver.hybrj(func2, jac2, np.array([y0, y1]),
#                                             np.array([-10, -10]), np.array([100, 100]))
#                    if (info == 1 and sol[0] != y0 and sol[1] != y1):
#                        # compute 3rd parameter
#                        N0 = k1 / (dD * np.nansum(D**sol[0] * np.exp(-sol[1]*D) * sigma1))
#                        return (N0, *sol)

#    # solve 2-parameter problem
#    x0s = np.arange(-2, 100, 0.2)
#    x1s = np.arange(0, 100, 0.2)
#    for x0 in x0s:
##        x1cent = 1.8e-6*x0**4 - 6.3e-4*x0**3 + 6.0e-2*x0**2 + 4.8e-1*x0 + 2.53
##        x1pos = np.linspace(x1cent, x1cent + 9.8, num=50)
##        x1neg = np.linspace(x1cent-0.2, x1cent - 10, num=49)
##        x1s = np.empty((len(x1pos) + len(x1neg),))
##        x1s[::2] = x1pos
##        x1s[1::2] = x1neg
#        for x1 in x1s:
#            sol, info = solver.hybrj(func2, jac2, np.array([x0, x1]),
#                                     np.array([-2, 0]), np.array([100, 100]))
#            if (info == 1 and sol[0] != x0 and sol[1] != x1):
#                # compute 3rd parameter
#                N0 = k1 / (dD * np.nansum(D**sol[0] * np.exp(-sol[1]*D) * sigma1))
#                return (N0, *sol)
#    else:
#        return np.nan, np.nan, np.nan


def get_gamDSD(ND, D, dD):
    # inverse gamma model (method of moments)
    M3 = uf.nansum(ND * D**3, dim='diameter') * dD
    M4 = uf.nansum(ND * D**4, dim='diameter') * dD
    M6 = uf.nansum(ND * D**6, dim='diameter') * dD

    G = uf.nandiv(M4**3, M6 * M3**2)
    mu = uf.nandiv(5.5 * G - 4 + np.sqrt(G * (G * 0.25 + 2)), 1 - G)
    lam = (mu + 4) * uf.nandiv(M3, M4)
    with np.errstate(over='ignore', invalid='ignore'):
        N_0 = uf.nandiv(lam**(mu+4), uf.gamma(mu+4))*M3

    N_0 = uf.where(mu < -1, np.nan, N_0)
    lam = uf.where(mu < -1, np.nan, lam)
    mu = uf.where(mu < -1, np.nan, mu)

    N_0.setname('gamma_dsd_N_0')
    lam.setname('gamma_dsd_lambda')
    mu.setname('gamma_dsd_mu')
    return N_0, mu, lam
