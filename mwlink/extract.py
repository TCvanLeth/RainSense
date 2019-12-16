#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:39:32 2017

@author: tcvanleth
"""

import numpy as np

from mwlink import inout as io
#
#data = io.import_ds('pars_l1', 'WURex14')
#f1 = data['forum_1', 'channel_1', 'precip_old']
#np.savetxt('test.txt', np.transpose([f1['time'].values, f1.values]))

data = io.import_ds('link_l2', 'WURex14', 'link_test6_1')
aux = io.import_ds('link_aux', 'WURex14', 'Leijnse_test6')
f1 = data['nokia', 'channel_1', 'precip_mean'].sel(time=slice('2015-03-11', '2016-01-04'))
#f1[aux['parsivel', 'channel_1', 'h_comp'].sel(htype='rain')==0] = 0
time = (f1['time'].astype(int) / 1000000).values
np.savetxt('nokia_link.txt', np.transpose([time, f1.values]), header='time link_precip',
           comments='')