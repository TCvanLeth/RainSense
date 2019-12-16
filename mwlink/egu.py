# -*- coding: utf-8 -*-

import phad as ph
import numpy as np
import matplotlib.pyplot as pl
import itertools
from scipy.optimize import curve_fit

import mwlink.inout as io


data0 = io.import_ds('pars_l1', 'WURex14')
data4 = io.import_ds('pars_l2', 'WURex14', 'Leijnse_test4')
data5 = io.import_ds('pars_l2', 'WURex14', 'Leijnse_test5')
data6 = io.import_ds('pars_l2', 'WURex14', 'Leijnse_test6')

R0 = data0[..., 'channel_1', 'precip_old']
R4 = data4[..., 'channel_1', 'precip']
R5 = data5[..., 'channel_1', 'precip']
R6 = data6[..., 'channel_1', 'precip']
k4 = data4.sel(frequency=38e9, polarization='H', how='nearest')[..., 'channel_1', 'k']
k5 = data5.sel(frequency=38e9, polarization='H', how='nearest')[..., 'channel_1', 'k']
k6 = data6.sel(frequency=38e9, polarization='H', how='nearest')[..., 'channel_1', 'k']
rain = data5.sel(htype='rain')[..., 'channel_1', 'h_comp']


def scatplot(x, y, name1, name2):
    ax = ph.plotting.plot(x, y, alpha=0.2)
    limit = float(min(ph.ufuncs.nanmax(x), ph.ufuncs.nanmax(y)).values)
    ax.plot([0, limit], [0, limit], 'k--')
    r2 = ph.analysis.corr(x, y)[0, 1]
    ax.text(0., 0.9*limit, 'r2 = %.3f' % r2)
    ax.set_xlabel('%s precipitation intensity [$mm/hr$]' % name1)
    ax.set_ylabel('%s precipitation intensity [$mm/hr$]' % name2)
    return r2


def correlogram(Rs, atts):
    def func(x, d0, s0):
        return np.exp(-(x/d0)**s0)

    L = np.zeros(10)
    r2 = np.zeros(10)
    for i, value in enumerate(zip(Rs, atts)):
        a = value[0][0]
        b = value[0][1]

        x1 = value[1][0]['site_longitude']
        y1 = value[1][0]['site_latitude']
        z1 = value[1][0]['site_altitude']
        x2 = value[1][1]['site_longitude']
        y2 = value[1][1]['site_latitude']
        z2 = value[1][1]['site_altitude']
        name1 = value[1][0]['station_id']
        name2 = value[1][1]['station_id']

        l = ph.geometry.haversines(x1, y1, z1, x2, y2, z2)
        L[i] = np.sqrt(l @ l)
        r2[i] = scatplot(a, b, name1, name2)

    par, cov = curve_fit(func, L, r2, p0=(3000, 1))
    y = func(L, par[0], par[1])

    indexer = np.argsort(L)
    limitx = float(np.nanmax(L))
    limity = float(np.nanmax(y))

    fig, ax = pl.subplots()
    ax.plot(L, r2, 'o')
    ax.plot(L[indexer], y[indexer], '-')
    ax.text(0.8*limitx, 0.9*limity, '$d_0=%.0f$\n$s_0=%.3f$' % (par[0], par[1]))
    return L, r2, par

R0f1, R0f2, R0bo, R0nv, R0bi = ph.ufuncs.broadcast(R0['forum_1'],
                                                   R0['forum_2'],
                                                   R0['bongerd'],
                                                   R0['nvwa'],
                                                   R0['biotechnion'])

R5f1, R5f2, R5bo, R5nv, R5bi = ph.ufuncs.broadcast(R5['forum_1'],
                                                   R5['forum_2'],
                                                   R5['bongerd'],
                                                   R5['nvwa'],
                                                   R5['biotechnion'])

R4f1, R4f2, R4bo, R4nv, R4bi = ph.ufuncs.broadcast(R4['forum_1'],
                                                   R4['forum_2'],
                                                   R4['bongerd'],
                                                   R4['nvwa'],
                                                   R4['biotechnion'])

R6f1, R6f2, R6bo, R6nv, R6bi = ph.ufuncs.broadcast(R6['forum_1'],
                                                   R6['forum_2'],
                                                   R6['bongerd'],
                                                   R6['nvwa'],
                                                   R6['biotechnion'])

R0f1 = R0f1[rain['forum_1']==1]
R0f2 = R0f2[rain['forum_2']==1]
R0bo = R0bo[rain['bongerd']==1]
R0nv = R0nv[rain['nvwa']==1]
R0bi = R0bi[rain['biotechnion']==1]

R4f1 = R4f1[rain['forum_1']==1]
R4f2 = R4f2[rain['forum_2']==1]
R4bo = R4bo[rain['bongerd']==1]
R4nv = R4nv[rain['nvwa']==1]
R4bi = R4bi[rain['biotechnion']==1]

R5f1 = R5f1[rain['forum_1']==1]
R5f2 = R5f2[rain['forum_2']==1]
R5bo = R5bo[rain['bongerd']==1]
R5nv = R5nv[rain['nvwa']==1]
R5bi = R5bi[rain['biotechnion']==1]

R6f1 = R6f1[rain['forum_1']==1]
R6f2 = R6f2[rain['forum_2']==1]
R6bo = R6bo[rain['bongerd']==1]
R6nv = R6nv[rain['nvwa']==1]
R6bi = R6bi[rain['biotechnion']==1]


Rs0 = list(itertools.combinations((R0f1, R0f2, R0bo, R0nv, R0bi), 2))
Rs4 = list(itertools.combinations((R4f1, R4f2, R4bo, R4nv, R4bi), 2))
Rs5 = list(itertools.combinations((R5f1, R5f2, R5bo, R5nv, R5bi), 2))
Rs6 = list(itertools.combinations((R6f1, R6f2, R6bo, R6nv, R6bi), 2))

atf1 = data0['forum_1'].attrs
atf2 = data0['forum_2'].attrs
atbo = data0['bongerd'].attrs
atnv = data0['nvwa'].attrs
atbi = data0['biotechnion'].attrs
atts = list(itertools.combinations((atf1, atf2, atbo, atnv, atbi), 2))

correlogram(Rs0, atts)
correlogram(Rs4, atts)
correlogram(Rs5, atts)

ax = ph.plotting.plot(R0['forum_1'], R5['forum_1'][rain['forum_1']==1], alpha=0.02)
ph.plotting.plot(R0['forum_1'], R4['forum_1'][rain['forum_1']==1], alpha=0.02, ax=ax)
ph.plotting.plot(R0['forum_1'], R6['forum_1'][rain['forum_1']==1], alpha=0.02, ax=ax)

ax = ph.plotting.plot(R4['forum_1'], R6['forum_1'][rain['forum_1']==1], alpha=0.02)

ax=ph.plotting.plot(k6['forum_1'], R6['forum_1'][rain['forum_1']==1], alpha=0.05)
ph.plotting.plot(k4['forum_1'], R4['forum_1'][rain['forum_1']==1], alpha=0.05, ax=ax)

R6f1 = R6f1[~ph.ufuncs.isnan(R6f1) & ~ph.ufuncs.isnan(R6f2) & ~ph.ufuncs.isnan(R6bo) &
            ~ph.ufuncs.isnan(R6nv) & ~ph.ufuncs.isnan(R6bi)]
R6f2 = R6f2[~ph.ufuncs.isnan(R6f1) & ~ph.ufuncs.isnan(R6f2) & ~ph.ufuncs.isnan(R6bo) &
            ~ph.ufuncs.isnan(R6nv) & ~ph.ufuncs.isnan(R6bi)]
R6bo = R6bo[~ph.ufuncs.isnan(R6f1) & ~ph.ufuncs.isnan(R6f2) & ~ph.ufuncs.isnan(R6bo) &
            ~ph.ufuncs.isnan(R6nv) & ~ph.ufuncs.isnan(R6bi)]
R6nv = R6nv[~ph.ufuncs.isnan(R6f1) & ~ph.ufuncs.isnan(R6f2) & ~ph.ufuncs.isnan(R6bo) &
            ~ph.ufuncs.isnan(R6nv) & ~ph.ufuncs.isnan(R6bi)]
R6bi = R6bi[~ph.ufuncs.isnan(R6f1) & ~ph.ufuncs.isnan(R6f2) & ~ph.ufuncs.isnan(R6bo) &
            ~ph.ufuncs.isnan(R6nv) & ~ph.ufuncs.isnan(R6bi)]
pluvio = pluvio[~ph.ufuncs.isnan(R6f1) & ~ph.ufuncs.isnan(R6f2) & ~ph.ufuncs.isnan(R6bo) &
            ~ph.ufuncs.isnan(R6nv) & ~ph.ufuncs.isnan(R6bi)]

ax = ph.ufuncs.nancumsum(R6f1/120, dim='time').plot()
ph.ufuncs.nancumsum(R6f2/120, dim='time').plot(ax=ax)
ph.ufuncs.nancumsum(R6bo/120, dim='time').plot(ax=ax)
ph.ufuncs.nancumsum(R6nv/120, dim='time').plot(ax=ax)
ph.ufuncs.nancumsum(R6bi/120, dim='time').plot(ax=ax)

###############################################################################
ND1 = ND[(p<1) & (rain['forum_1']==1)]
pdf = ph.ufuncs.nansum(ND1, dim='time') / ph.ufuncs.nansum(ND1, dim=['time', 'diameter'])
ax = pdf.plot()
pdf.barplot(ax=ax, alpha=0.25)
ND1 = ND[(p >= 1) & (p < 5) & (rain['forum_1']==1)]
pdf = ph.ufuncs.nansum(ND1, dim='time') / ph.ufuncs.nansum(ND1, dim=['time', 'diameter'])
pdf.plot(ax=ax)
pdf.barplot(ax=ax, alpha=0.25)
ND1 = ND[(p >= 5) & (p < 25) & (rain['forum_1']==1)]
pdf = ph.ufuncs.nansum(ND1, dim='time') / ph.ufuncs.nansum(ND1, dim=['time', 'diameter'])
pdf.plot(ax=ax)
pdf.barplot(ax=ax, alpha=0.25)
ND1 = ND[(p >= 25) & (rain['forum_1']==1)]
pdf = ph.ufuncs.nansum(ND1, dim='time') / ph.ufuncs.nansum(ND1, dim=['time', 'diameter'])
pdf.plot(ax=ax)
pdf.barplot(ax=ax, alpha=0.25)

ax = ph.plotting.plot(R5f1[rain['forum_1']==1], R5f2[rain['forum_2']==1], alpha=0.2)
ph.analysis.corr(R5f1[rain['forum_1']==1], R5f2[rain['forum_2']==1])

ND = data5[..., 'channel_1', 'N(D)']
NDf1, NDf2, NDbo, NDnv, NDbi = ph.ufuncs.broadcast(ND['forum_1'],
                                                   ND['forum_2'],
                                                   ND['bongerd'],
                                                   ND['nvwa'],
                                                   ND['biotechnion'])

M0f1 = (NDf1 * NDf1['diameter_binwidth']).sum(dim='diameter')[rain['forum_1']==1]
M0f2 = (NDf2 * NDf2['diameter_binwidth']).sum(dim='diameter')[rain['forum_2']==1]
M0bo = (NDbo * NDbo['diameter_binwidth']).sum(dim='diameter')[rain['bongerd']==1]
M0nv = (NDnv * NDnv['diameter_binwidth']).sum(dim='diameter')[rain['nvwa']==1]
M0bi = (NDbi * NDbi['diameter_binwidth']).sum(dim='diameter')[rain['biotechnion']==1]

scatplot(M0f1, M0f2, 'Forum_1', 'Forum_2')
scatplot(M0f1, M0bo, 'Forum_1', 'Bongerd')
scatplot(M0f1, M0nv, 'Forum_1', 'NVWA')
scatplot(M0f1, M0bi, 'Forum_1', 'Biotechnion')
