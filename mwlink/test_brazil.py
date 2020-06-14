#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:38:15 2017

@author: tcvanleth
"""
import numpy as np
import pyhad as ph

from mwlink import inout as io


data2 = io.import_ds('pars_l2', 'Brazil', proID='test1')
data1 = io.import_ds('pars_l2', 'Brazil', proID='test2')
l1dat = io.import_ds('pars_l1', 'Brazil')

rain = data1['ieav', 'channel_1', 'h_comp'].sel(htype='rain')
p0 = l1dat['ieav', 'channel_1', 'precip_old'][rain == True]
p1 = data1['ieav', 'channel_1', 'precip'][rain == True]
p2 = data2['ieav', 'channel_1', 'precip'][rain == True]

k1 = data1['ieav', 'channel_1', 'k'].sel(frequency=38e9, polarization='V')[rain == True]
k2 = data2['ieav', 'channel_1', 'k'].sel(frequency=38e9, polarization='V')[rain == True]

x = np.arange(0, 10, 0.1)
y1 = 9.17 * x**1.042
y2 = 10.29 * x**1.044

ax = ph.plotting.plot(k2, p2)
