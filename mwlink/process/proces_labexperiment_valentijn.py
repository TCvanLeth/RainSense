#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:16:38 2018

@author: tcvanleth
"""

import logging

import pandas as pd
import pyhad as ph

from mwlink import inout as io
from mwlink.process.disdro import disdro_level2 as l2


def proces_wurex_pars(proID, setID='WURex14', begin=None, end=None):
    logging.info('processing parsivel data')

    l1adat = io.import_ds('pars_l1', setID, conform=True)
    l2dat = l2.disdro_l2(l1adat)
    io.export_ds(l2dat, level='pars_l2', pro_id=proID)


if __name__ == '__main__':
    ph.common.standardlogger()
    proID = 'htype_algo_test2_thurai_interp3_2'
    proces_wurex_pars(proID, setID='labexperiment_valentijn2')

    data = io.import_ds('pars_l2', 'labexperiment_valentijn2', pro_id=proID)['forum_1', 'channel_1']
    R = data['precip'].values
    time = data['time'].values
    df = pd.DataFrame(R, index=time, columns=['rain intensity [mm/h]'])
    df.to_csv('/home/tcvanleth/parsivel_2018_06_12.dat')