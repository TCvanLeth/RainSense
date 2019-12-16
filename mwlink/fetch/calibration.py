#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:20:41 2017

@author: tcvanleth
"""

import os

import numpy as np
import matplotlib.pyplot as pl
import phad as ph
from scipy.stats import linregress

from mwlink import settings


power = []
voltage = []

path = "/home/tcvanleth/Data/WURex14/Calibration/nokia_calibration.dat"
with open(path) as inh:
    lines = [line.strip('\r\n').split('\t') for line in inh]
power.append(np.asarray([float(line[1]) for line in lines]))
voltage.append(np.asarray([float(line[2]) for line in lines]))

path = "/home/tcvanleth/Data/WURex14/Calibration/RAL_38H_calibration.dat"
with open(path) as inh:
    lines = [line.strip('\r\n').split('\t') for line in inh]
power.append(np.asarray([float(line[0]) for line in lines]))
voltage.append(np.asarray([float(line[2]) for line in lines]))

path = "/home/tcvanleth/Data/WURex14/Calibration/RAL_38V_calibration.dat"
with open(path) as inh:
    lines = [line.strip('\r\n').split('\t') for line in inh]
power.append(np.asarray([float(line[0]) for line in lines]))
voltage.append(np.asarray([float(line[2]) for line in lines]))

path = "/home/tcvanleth/Data/WURex14/Calibration/RAL_26_calibration.dat"
with open(path) as inh:
    lines = [line.strip('\r\n').split('\t') for line in inh]
power.append(np.asarray([float(line[0]) for line in lines]))
voltage.append(np.asarray([float(line[2]) for line in lines]))


fig, axes = pl.subplots(1,4, figsize=(16, 4))
for i, ax in enumerate(axes):
    volt = voltage[i]
    powr = power[i
                 ]
    ax.plot(volt, powr, 'o')
    slope, intercept, r, p, stderr = linregress(volt, powr)
    voltrange = np.linspace(volt.min(), volt.max(), 1000)
    powerhat = intercept + slope * voltrange
    ax.plot(voltrange, powerhat, color='#262626')
    ax.set_xlabel('voltage [V]')
    ax.set_ylabel('power [dBm]')
    ax.text(0.2, 0.1, '$y=%.3f+%.3f x$\n$r^{2}=%.3f$' % (intercept, slope, r**2),
            transform=ax.transAxes)
pl.tight_layout()
ph.plotting.sublabels(axes)

figpath = os.path.join(settings.plotpath, 'WURex14/link_test8')
if not os.path.exists(figpath):
    os.makedirs(figpath)

fig.savefig(os.path.join(figpath, 'calibration.pdf'), dpi=300)
fig.savefig(os.path.join(figpath, 'calibration.png'), dpi=300)