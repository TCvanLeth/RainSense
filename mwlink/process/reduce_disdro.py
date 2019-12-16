#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:45:34 2019

@author: tcvanleth
"""

import logging

import phad as ph

from mwlink import inout as io
from mwlink.process.link import link_level2 as l2


def proc_link_l2(setID, DSD_setID, proID, DSD_proID, **kwargs):
    inlvl = 'pars_l1'

    logging.info('processing disdro data')
    data = io.import_ds(inlvl, setID, **kwargs)
    data = data.select(system_type='disdrometer', quantity='particle_count')

    io.export_ds(data, level='pars_l2', pro_id=proID)


if __name__ == '__main__':
    ph.common.standardlogger()

    setID = 'WURex14'
    DSD_setID = setID
    proID = 'pars_reduced'
    DSD_proID = 'new_test1'
    proc_link_l2(setID, DSD_setID, proID, DSD_proID, times=slice('2014-08-21', '2016-01-01'))
