# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:00:23 2015

@author: T.C. van Leth
"""
# TO-DO: plot data
# TO-DO: analyze data
# TO-DO: link downsampled set to larger set

from . import inout_links as io
from . import plotting


def pull(link_ID, pro_ID, links=[], times=None, sel={}, where={}):
    data = io.import_ls(link_ID, pro_ID=pro_ID, times=times, links=links)
    return data.where(**where).select(**sel)


def linkseries(plotname=None):
    data = pull('WURex14', 'testB1',
                sel={'quantity': 'specific_attenuation', 'sampling': 'mean'})
    dlist = {'RAL_38_V': data.links['RAL_38'].channels['channel_1']['k_mean'],
             'RAL_38_H': data.links['RAL_38'].channels['channel_2']['k_mean'],
             'RAL_26': data.links['RAL_26'].channels['channel_1']['k_mean'],
             'Nokia': data.links['Nokia'].channels['channel_1']['k_mean'],
             'Scintec_mod_1750': data.links['Scintec'].channels['channel_1']['k_mean'],
             'Scintec_mod_2500': data.links['Scintec'].channels['channel_2']['k_mean']}

    fig = plotting.timeseries(dlist)
    if plotname:
        plotting.save_plot(fig, plotname)


def pull_scatter(plotname=None):
    data = pull('WURex14', 'test1', sel={'sampling': 'mean'})
    x = data.links['RAL_38'].channels['auxiliary']['temp_Rx']
    y = data.links['RAL_38'].channels['channel_1']['k_mean']
    fig = plotting.scatter(x, y)
    if plotname:
        plotting.save_plot(fig, plotname)


def rainseries(plotname=None):
    disdro = pull('link_aux', 'WURex14', 'testA1',
                  sel={'quantity': 'rain_rate', 'sampling': 'mean'})
    linkdat = pull('link_l2', 'WURex14', 'testB1',
                   sel={'quantity': 'rain_rate', 'sampling': 'mean'})

    dlist = {'parsivel': disdro.links['parsivel'].channels['auxiliary']['R_new'],
             'Nokia': linkdat.links['Nokia'].channels['channel_1']['R'],
             'RAL_38_V': linkdat.links['RAL_38'].channels['channel_1']['R'],
             'RAL_38_H': linkdat.links['RAL_38'].channels['channel_2']['R'],
             'RAL_38_dpa': linkdat.links['RAL_38'].channels['channel_5']['R'],
             'RAL_26': linkdat.links['RAL_26'].channels['channel_1']['R']}
    fig = plotting.timeseries(dlist)
    if plotname:
        plotting.save_plot(fig, plotname)
