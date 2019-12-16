#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:53:51 2017

@author: tcvanleth
"""

import os

import numpy as np
import phad as ph

import mwlink.inout as io


def read(name):
    indir = '/home/tcvanleth/Data/cross_sections_Hidde'
    inpath = os.path.join(indir, name)
    with open(inpath, 'r') as inh:
        rows = [np.array(line.strip(' \r\n').split(' ')) for line in inh]
    return np.stack(rows).squeeze().astype(float)

D = ph.Index(read('T_matrix_LUT_D.txt'), 'diameter')
D.setattrs(quantity='diameter', unit='mm')
f = ph.Index(read('T_matrix_LUT_f.txt'), 'frequency')
f.setattrs(quantity='frequency', unit='GHz')
pol = ph.Index(['H', 'V'], 'polarization')

Qhh = read('T_matrix_LUT_Qext_hh.txt')
Qvv = read('T_matrix_LUT_Qext_vv.txt')
Q = ph.Array(np.stack([Qhh, Qvv]), coords=[pol, D, f], name='ext')
Q.setattrs(name='extinction_cross_section', quantity='area', unit='mm2')
io.export_ds(ph.Channel([Q], name='simulated'), level='scatter', pro_id='Leijnse')
