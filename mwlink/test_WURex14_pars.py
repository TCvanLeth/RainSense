#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:14:46 2017

@author: tcvanleth
"""

import numpy as np
import pyhad as ph

from mwlink import inout as io


data2 = io.import_ds('pars_l2', 'WURex14', proID='pars_test12')
data1 = io.import_ds('pars_l2', 'WURex14', proID='pars_test11')
l1dat = io.import_ds('pars_l1', 'WURex14')

rain = data2['forum_2', 'channel_1', 'h_comp'].sel(htype='rain')
p0 = l1dat['forum_2', 'channel_1', 'precip_old'][rain == True]
p1 = data1['forum_2', 'channel_1', 'precip'][rain == True]
p2 = data2['forum_2', 'channel_1', 'precip'][rain == True]

k1 = data1['forum_2', 'channel_1', 'k'].sel(frequency=38e9, polarization='V')[rain == True]
k2 = data2['forum_2', 'channel_1', 'k'].sel(frequency=38e9, polarization='V')[rain == True]

x = np.arange(0, 10, 0.1)
y1 = 9.17 * x**1.042
y2 = 6.25 * x**0.838

ax = ph.plotting.plot(k2, p2)