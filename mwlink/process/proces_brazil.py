#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:22:45 2017

@author: tcvanleth
"""

import phad as ph

from mwlink import inout as io
from mwlink.process.atmo_level2 import atmo_l2
from mwlink.process.disdro import disdro_level1 as l1
from mwlink.process.disdro import disdro_level2 as l2


def proces_brazil():
    setID = 'Brazil'
    proID = 'Leijnse_test1'
    scatID = 'Leijnse'

    l1adat = io.import_ds('pars_l1', setID, conform=True)
    scat = io.import_ds('scatter', 'simulated', scatID, conform=True)

    P = ph.Array(1e5, name='P')
    P.setattrs(quantity='pressure', unit='Pa')
    T = ph.Array(25.1 + 273.15, name='T')
    T.setattrs(quantity='temperature', unit='K', wet=False, ventilated=True)
    atmdat = atmo_l2(ph.Channel([P, T]), bins=l1adat['ieav', 'channel_1', 'diameter'])

    l1adat = l1.precip_conv(l1adat)
    l1adat = l1.pars_correction(l1adat, atmdat)
    scat = scat.reindex(temperature=T, diameter=l1adat['ieav', 'channel_1', 'diameter'],
                        how='nearest')

    l2dat = l2.disdro_l2(l1adat, scat)
    l2dat = l2dat.merge(l1adat.aselect(quantity='particle_count'))
    io.export_ds(l2dat, level='pars_l2', pro_id=proID)

if __name__ == "__main__":
    ph.common.standardlogger()
    proces_brazil()
