# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 16:47:10 2015

@author: T.C. van Leth

This script converts the ASCII datasheets into hdf files for easier access
"""

import os
import re
import datetime as dt
import logging

from dask import delayed, compute
import dask.array as da
import numpy as np
import phad as ph
import yaml

from mwlink import inout as io
from mwlink.fetch import fetch_pars_common as fp


def fetch_disdro(indir, begin=None, end=None, mfile=None):
    """
    read the parsivels telegrams and convert to DSh5 format.
    """
    # fetch parameters
    if mfile is None:
        mfile = 'parsivels.yaml'
    metapath = os.path.join(os.path.dirname(__file__), mfile)
    with open(metapath, mode='r') as inh:
        meta = yaml.safe_load(inh)

    parpath = os.path.join(os.path.dirname(__file__), 'devices.yaml')
    with open(parpath, mode='r') as inh:
        device = yaml.safe_load(inh)['OTT']['Parsivel_1']
    scales = fp.create_bins(device)

    # define experiment period
    if begin is None:
        begin = '2014-06-20'
    if end is None:
        end = dt.date.today()
    begin = np.datetime64(begin)
    end = np.datetime64(end)
    dates = np.arange(begin, end)
    dates = np.core.defchararray.replace(dates.astype(str), '-', '_')

    # list input files
    files = [os.path.join(p, f) for p, _, fs in os.walk(indir) for f in fs]
    if len(files) == 0:
        return

    logging.info('converting parsivel telegrams to hdf5 format')
    pattern = re.compile(r'''((?:[^,"']|"[^"]*"|'[^']*')+)''')
    ntime = {}
    start = {}
    stop = {}
    for sID, smet in meta['stations'].items():
        infiles = [x for x in files if smet['prefix'] in x]
        if not infiles:
            continue

        if smet['format'] == 'CR1000':
            sfunc = lambda x: [line.split(",") for line in x[4:]]
            time_col = 0
        else:
            sfunc = lambda x: [pattern.split(line)[1::2] for line in x]
            time_col = 3

        for date in dates:
            dfiles = [x for x in infiles if date in x]
            if not dfiles:
                continue
            for inpath in dfiles:
                if os.stat(inpath).st_size == 0:
                    continue
                ntime[inpath], start[inpath], stop[inpath] = getlength(inpath,
                                                                       time_col,
                                                                       sfunc)
    mdat = []
    for sID, smet in meta['stations'].items():
        infiles = [x for x in files if smet['prefix'] in x]
        if not infiles:
            continue
        dtypes, dims, headers, shape = fp.prepare_dicts(sID, smet, device)
        size = np.multiply(*shape.values())
        if smet['format'] == 'CR1000':
            sfunc = lambda x: [line.split(",") for line in x[4:]]
            time_col = 0
            rfunc = readfunc_pars
        else:
            sfunc = lambda x: [pattern.split(line)[1::2] for line in x]
            time_col = 3
            rfunc = readfunc_swiss

        for date in dates:
            dfiles = [x for x in infiles if date in x]
            if not dfiles:
                continue
            for inpath in dfiles:
                if os.stat(inpath).st_size == 0:
                    continue
                if ntime[inpath] == 0:
                    continue
                data = read(inpath, time_col, headers, dims, dtypes, rfunc,
                            sfunc, shape, size)

                logging.info('converting %s' % (inpath))
                time = da.from_delayed(data['time'], (ntime[inpath],), '<M8[us]')
                chunks = len(time) * 2
                time = ph.TimeScale(time, chunksize=chunks,
                                    start=start[inpath], stop=stop[inpath])
                time.step = None
                step = meta['temporal_resolution']
                scales['time'] = time
                data = ph.convert_to_set(data, scales, meta, chunksize=chunks)

                logging.info('regularizing %s' % (inpath))
                data = ph.regularize(data, [scales['time']], [step])
                mdat.append(data)
    if mdat:
        logging.info('merging data')
        mdat = ph.merge(mdat)
        io.export_ds(mdat)
    logging.info('finished converting parsivel files')


def read(inpath, time_col, headers, dims, dtypes, rfunc, sfunc, shape, size):
    logging.info('reading %s' % (inpath))
    rows = read_lines(inpath, sfunc)
    times = get_time(rows, time_col)

    where = fp.get_where(times)
    data = {}
    data['time'] = times[where]
    for name, header in headers.items():
        dim = dims[name]
        shp = [shape[x] for x in dim if x != 'time']
        dtype = dtypes[name]
        idat = rfunc(rows, dim, header, dtype, size, where)
        data[name] = fp.convert(idat, dtype, shp)
    return data


def getlength(inpath, time_col, sfunc):
    logging.info('reading %s' % (inpath))
    rows = read_lines(inpath, sfunc)
    times = get_time(rows, time_col)

    where = fp.get_where(times)
    where, times = compute(where, times)
    ndim = len(where)
    if ndim > 0:
        return ndim, times[0], times[-1] + 1
    else:
        return ndim, None, None


###############################################################################
@delayed(pure=True)
def read_lines(inpath, sfunc):
    with open(inpath, 'r') as inh:
        lines = (line.rstrip() for line in inh)
        lines = (line for line in lines if line)
        lines = [line.strip('\r\n') for line in lines]
        rows = sfunc(lines)
    return rows


@delayed(pure=True)
def get_time(rows, time_col):
    times = np.asarray([x[time_col] for x in rows], dtype=str)
    return np.char.strip(times, '"').astype('<M8[us]')


@delayed(pure=True)
def readfunc_pars(rows, dim, header, dtype, size, where=None):
    idat = []
    for i, row in enumerate(rows):
        if where is not None and i not in where:
            continue

        # parse telegram
        meas = re.split(r'\s\s(?=\d\d:)', row[2].strip('"'))
        numbers = [x[:3] for x in meas]
        row = [x[3:] for x in meas]

        # check if entry exists
        index = header
        if index in numbers:
            col = row[numbers.index(index)]
        else:
            col = ''

        if len(dim) == 3:
            classes = col[:-1].split(';')
            if len(classes) != size:
                idat += [[ph.common.get_fill(dtype)] * size]
            else:
                idat += [classes]
        else:
            if col == '' or col == 'na' or ';' in col:
                idat += [ph.common.get_fill(dtype)]
            else:
                idat += [col]
    return idat


@delayed(pure=True)
def readfunc_swiss(rows, dim, header, dtype, size, where=None):
    idat = []
    for i, row in enumerate(rows):
        if where is not None and i not in where:
            continue

        index = header
        if np.isnan(index):
            idat += [ph.common.get_fill(dtype)]
            continue

        # check if entry exists
        if index >= len(row):
            col = ''
        else:
            col = row[index].strip('"')

        # read entries
        if len(dim) == 3:
            classes = col[:-1].split(',')
            if len(classes) != size:
                idat += [[ph.common.get_fill(dtype)] * size]
            else:
                idat += [classes]
        else:
            if col == '' or col == 'na' or ',' in col:
                idat += [ph.common.get_fill(dtype)]
            else:
                idat += [col]
    return idat


if __name__ == "__main__":
    ph.common.standardlogger()
    #from dask.cache import Cache
    #with Cache(2e8) as cache:
#    path = "/home/tcvanleth/Data/WURex14/Unprocessed_data"
#    times = ['2014-08-20', '2014-12-01', '2015-04-01', '2015-06-01', '2015-08-01',
#             '2015-10-01', '2016-02-01']
#    for i in range(2, len(times) - 1):
#        fetch_disdro(path, begin=times[i], end=times[i+1])
#
#    dat = []
#    path = '/home/tcvanleth/Data2/WURex14/WURex14_pars_l1'
#    dat.append(ha.inout_common.from_hdf(ha.Network, path, hi_res=True))
#    for i in range(1, len(times)-1):
#        path = '/home/tcvanleth/Data2/WURex14/WURex14_pars_l1_'+str(i)
#        dat.append(ha.inout_common.from_hdf(ha.Network, path, hi_res=True))
#
#    outpath = '/home/tcvanleth/Data2/WURex14/WURex14_pars_l1_comb'
#    cdat = ha.Network.merged(dat)
#    cdat.store(outpath)

    path = "/home/tcvanleth/Data/labexperiment_valentijn2"
    fetch_disdro(path, mfile='parsivel_valentijn.yaml', begin='2018-06-12', end='2018-06-26')
