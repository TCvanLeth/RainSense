# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:22:11 2015

@author: T.C. van Leth
"""


def maybe_resample(link, mattrs):
    pass
    return link


def minmax_to_mean(link):
    alpha = 0.244
    maxR = link['rain_max']
    minR = link['rain_min']

    mean = alpha*maxR+(1-alpha)*minR
    mean.attrs = minR.attrs
    mean.attrs['sampling'] = 'mean'

    # repack data
    link.drop('rain_max').drop('rain_min')
    link['rain'] = mean
    return link
