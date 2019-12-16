# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:43:13 2015

@author: T.C. van Leth

this code computes several derived quantities from the corrected Parsivel DSD
TO-DO: provision for missing veenkampen data
"""
import logging

import phad as ph

from mwlink import inout as io
from mwlink.process.disdro import disdro_level1 as l1
from mwlink.process.disdro import disdro_level2 as l2


def proces_wurex_pars(proID, begin=None, end=None):
    logging.info('processing parsivel data')
    setID = 'WURex14'
    scatID = 'thurai_full'
    auxID = 'Veenkampen'
    auxproID = 'atmo_test2'

    l1adat = io.import_ds('pars_l1', setID, conform=True)
    scat = io.import_ds('scatter', 'simulated', scatID, conform=True).sel(frequency=[26e9, 38e9])
    atmdat = io.import_ds('atmo_l2', auxID, auxproID, conform=True)
    atmdat = atmdat['veenkampen', 'channel_1']

    if begin is not None and end is not None:
        l1adat = l1adat.sel(time=slice(begin, end))
        atmdat = atmdat.sel(time=slice(begin, end))

    l1adat = l1.pars_correction(l1adat, atmdat)
#    io.export_ds(l1adat, level='pars_l2', pro_id=proID)
#    return

#    T = atmdat.aselect(quantity='temperature')
#    scat = scat.reindex(temperature=T, how='nearest')
    scat = scat.sel(temperature=288)
    del scat.attrs['level']
    del scat.attrs['pro_id']

    l2dat = l2.disdro_l2(l1adat, scat)
    io.export_ds(l2dat, level='pars_l2', pro_id=proID)


if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    #from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

    ph.common.standardlogger()
    proID = 'htype_algo_test2_thurai_interp2'
#    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler() as cprof:
#        proces_wurex_pars(proID, begin='2015-09-04', end='2015-09-05')

    times = np.arange(np.datetime64('2015-05-01'), np.datetime64('2016-06-01'),
                      np.timedelta64(1, 'D'))
#    for i in tqdm(range(len(times) - 1)):
#        proces_wurex_pars(proID, begin=times[i], end=times[i+1])
#    proces_wurex_pars(proID, begin='2015-04-01', end='2015-05-01')
    dat = []
    path = '/home/tcvanleth/Data2/WURex14/pars_l2/WURex14_pars_l2_' + proID
    dat.append(ph.inout_common.from_hdf(ph.Network, path)[:, :, ['N(D)',
               'precip', 'lwc', 'k', 'phi', 'gamm_dsd_N_0', 'gamm_dsd_lambda',
               'gamm_dsd_mu']])
    for i in tqdm(range(1, 60)):
        path2 = path+ '_' + str(i)
        dat.append(ph.inout_common.from_hdf(ph.Network, path2)[:, :, ['N(D)',
                   'precip', 'lwc', 'k', 'phi', 'gamm_dsd_N_0',
                   'gamm_dsd_lambda', 'gamm_dsd_mu']])
    outpath = path + '_comb4'
    print('merging')
    cdat = ph.Network.merged(dat)
    print('saving')
    cdat.store(outpath)

#    for j in tqdm(range(10, 274, 10)):
#        dat = []
#        for i in range(j, j+10):
#            path2 = path+ '_' + str(i)
#            dat.append(ha.inout_common.from_hdf(ha.Network, path2))
#        outpath = path + '_comb'
#        cdat = ha.Network.merged(dat)
#        cdat.store(outpath)

