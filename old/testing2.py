# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:06:05 2015

@author: T.C. van Leth
"""

from . import link_level1c as l1
from . import link_level2 as l2
from . import inout as io
from . import filehandler as fh
from . import plotting as pt
from . import dsd

datapath = 'D:/TmobileNL'
dirs = fh.dirstruct(datapath)

# load data
linkID = '4127A-4068B'
channelID = '4400-7496#'
l1bdat = io.hread('D:/TmobileNL/Processed/NEC_link_l1a', '/')
parsdat = io.extract(some_path, '/')
atmodat = io.extract(some_other_path, '/')

# convert raw signals to specific attenuation
l1cdat = l1bdat.apply(l1c_link)


# calculate rain rate
freq = l1cdat[ilink][ichan].attrs['frequency']
pol = l1cdat[ilink][ichan].attrs['polarisation']

DSDdat = dsd.DSDstats(parsdat, atmodat=atmodat)
k = DSDdat.atten(freq, pol)
R = DSDdat.rate()
powerlaw = l2.powerfit(k, R)[0]
l2dat = l2.get_rain(l1cdat, powerlaw)

# retrieve mean from max and min

# plot
pt.timeseries({linkID: l1cdat[linkID]}, var='k_max')
pt.freqplot({linkID: l1cdat[linkID]}, var='k_max')
