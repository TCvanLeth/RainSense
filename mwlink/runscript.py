#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:45:00 2017

@author: tcvanleth
"""

import phad as ph

from mwlink import inout as io
from mwlink.process.atmo_level2 import atmo_l2
from mwlink.process.disdro import disdro_level1 as l1
from mwlink.process.disdro import disdro_level2 as l2
from mwlink.process.disdro.disdro_resample import proc_path_int
from mwlink.process.powerlaws import proc_pwlaw
from mwlink.process.link.proces_link import proc_link_l2


ph.common.standardlogger()

setID = 'WURex14'
proID = 'Leijnse_test7'
scatID = 'Leijnse'
atmoID = 'Veenkampen'
atmoproID = 'atmo_test2'
linkproID = 'link_test7'

#data = io.import_ds('atmo_l1', atmoID, conform=True)
#l2dat = atmo_l2(data)
#io.export_ds(l2dat, level='atmo_l2', pro_id=atmoproID)

#l1adat = io.import_ds('pars_l1', setID, conform=True)
#scat = io.import_ds('scatter', 'simulated', scatID, conform=True)
#atmdat = io.import_ds('atmo_l2', atmoID, atmoproID, conform=True)
#atmdat = atmdat['veenkampen', 'channel_1']
#atmdat = atmdat.reindex(diameter=l1adat['forum_1', 'channel_1', 'diameter'],
#                        how='nearest')
#
#l1adat = l1.precip_conv(l1adat)
#l1adat = l1.pars_correction(l1adat, atmdat)
#
#scat = scat.reindex(diameter=l1adat['forum_1', 'channel_1', 'diameter'],
#                    how='nearest')
#del scat.attrs['level']
#del scat.attrs['pro_id']
#del scat.attrs['temporal_resolution']
#
#l2dat = l2.disdro_l2(l1adat, scat)
#l2dat = l2dat.merge(l1adat.aselect(quantity='particle_count'))
#io.export_ds(l2dat, level='pars_l2', pro_id=proID)
#
#proc_path_int(setID, proID)
proc_pwlaw(setID, proID)
proc_link_l2(setID, setID, linkproID, proID)