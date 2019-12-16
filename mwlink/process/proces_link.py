# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 08:28:43 2015

@author: T.C. van Leth
"""

import logging

import phad as ph

from mwlink import inout as io
from mwlink.process.link import link_level2 as l2


def proc_link_l2(setID, DSD_setID, proID, DSD_proID, **kwargs):
    inlvl = 'link_l1_comb'

    logging.info('processing link data')
    data = io.import_ds(inlvl, setID, conform=True, hi_res=True, **kwargs)
    data = data.select(system_type='microwave_link').rechunk(500000)
    data = data.resample('30s')
    aux = io.import_ds('link_aux', setID, DSD_proID, **kwargs)
    aux = aux['parsivel', 'channel_1'].rechunk(500000)

    powlaws = io.import_ds('powerlaw', 'WURex14', DSD_proID)['channel_1']
    calib = io.import_ds('calibration', 'mwlink_device_test')
    l1bdat = l2.link_l1b(data, calib)
    l1cdat = l2.link_l1c(l1bdat, aux)

    l2dat = l2.link_l2(l1cdat, powlaws)
    print(l2dat)
    raise Exception
    #mdat = ha.merge([l1bdat, l1cdat, l2dat])
    io.export_ds(l1cdat, level='link_l2', pro_id=proID)


if __name__ == '__main__':
    ph.common.standardlogger()

    setID = 'WURex14'
    DSD_setID = setID
    proID = 'link_test9'
    DSD_proID = 'new_test1'
    proc_link_l2(setID, DSD_setID, proID, DSD_proID, times=slice('2014-08-21', '2016-01-01'))
