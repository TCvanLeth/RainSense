#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:03:27 2017

@author: tcvanleth
"""

import pyhad as ph

from mwlink import inout as io
from mwlink.process.link import link_level2 as l2


ph.common.standardlogger()

setID = 'WURex14'
DSD_setID = setID
proID = 'link_test7'
DSD_proID = 'Leijnse_test7'
inlvl = 'link_l1'

data = io.import_ds(inlvl, setID, conform=True)
data = data.select(system_type='scintilometer').rechunk(100000)
calib = io.import_ds('calibration', 'mwlink_device_test')

l1bdat = data.apply(l2.l1b_chan, calib)
io.export_ds(l1bdat, level='scint_l2', pro_id=proID)
