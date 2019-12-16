#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:11:56 2017

@author: tcvanleth
"""

import phad as ph

dat = []
path = '/home/tcvanleth/Data2/Raupach_dsd_simulation/dsd2/Raupach_dsd_simulation_dsd2_test3'
#dat.append(ha.inout_common.from_hdf(ha.Channel, path))
for i in range(1, 11):
    path1 = path + '_'+str(i)
    dat.append(ph.inout_common.from_hdf(ph.Channel, path1))

outpath = path + '_comb'
cdat = ph.Channel.merged(dat)
cdat.store(outpath)