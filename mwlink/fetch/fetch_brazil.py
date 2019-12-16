#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:15:00 2017

@author: tcvanleth
"""

import datetime as dt
import logging
import os

from dask import delayed
import dask.array as da
from dask.delayed import compute
import numpy as np
import phad as ph
import yaml

from mwlink import inout as io
from mwlink import fetch_pars_common as fp


def fetch_disdro(indir, begin=None, end=None):
    """
    read the parsivels telegrams and convert to DSh5 format.
    """
    # fetch parameters
    metapath = os.path.join(os.path.dirname(__file__), 'brazil.yaml')
    with open(metapath, mode='r') as inh:
        meta = yaml.safe_load(inh)

    parpath = os.path.join(os.path.dirname(__file__), 'devices.yaml')
    with open(parpath, mode='r') as inh:
        device = yaml.safe_load(inh)['OTT']['Parsivel_1']
    scales = fp.create_bins(device)

    # define experiment period
    if begin is None:
        begin = '2014-08-20'
    if end is None:
        end = dt.date.today()
    begin = np.datetime64(begin)
    end = np.datetime64(end)
    dates = np.arange(begin, end)
    dates = np.core.defchararray.replace(dates.astype(str), '-', '')

    # list input files
    files = [os.path.join(p, f) for p, _, fs in os.walk(indir) for f in fs]
    if len(files) == 0:
        return

    logging.info('converting parsivel telegrams to hdf5 format')
    rfunc = readfunc_brazil
    ntime = {}
    start = {}
    stop = {}
    for sID, smet in meta['stations'].items():
        infiles = [x for x in files if smet['prefix'] in x]
        if not infiles:
            continue

        for date in dates:
            dfiles = [x for x in infiles if date in x]
            if not dfiles:
                continue
            for inpath in dfiles:
                ntime[inpath], start[inpath], stop[inpath] = getlength(inpath,
                                                                       rfunc)

    mdat = []
    for sID, smet in meta['stations'].items():
        infiles = [x for x in files if smet['prefix'] in x]
        if not infiles:
            continue
        dtypes, dims, headers, shape = fp.prepare_dicts(sID, smet, device)
        size = np.multiply(*shape)

        for date in dates:
            dfiles = [x for x in infiles if date in x]
            if not dfiles:
                continue
            for inpath in dfiles:
                if ntime[inpath] == 0:
                    continue
                data = read(inpath, headers, dims, dtypes, rfunc, shape, size)

                logging.info('converting %s' % (inpath))
                time = da.from_delayed(data['time'], (ntime[inpath],), '<M8[us]')
                time = ph.TimeScale(time, chunksize=len(time),
                                    start=start[inpath], stop=stop[inpath])
                time.step = None
                step = meta['temporal_resolution']
                scales['time'] = time
                data = ph.convert_to_set(data, scales, meta)

                logging.info('regularizing %s' % (inpath))
                data = ph.regularize(data, [scales['time']], [step])
                mdat.append(data)
    if mdat:
        logging.info('merging data')
        mdat = ph.merge(mdat)
        io.export_ds(mdat)
    logging.info('finished converting parsivel files')


def read(inpath, headers, dims, dtypes, rfunc, shape, size):
    logging.info('reading %s' % (inpath))
    rows = read_lines(inpath)
    time = rfunc(rows, ['time'], 20, str, 1)
    date = rfunc(rows, ['time'], 21, str, 1)
    times = get_time_pars(time, date)
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


def getlength(inpath, rfunc):
    logging.info('reading %s' % (inpath))
    rows = read_lines(inpath)
    time = rfunc(rows, ['time'], 20, str, 1)
    date = rfunc(rows, ['time'], 21, str, 1)
    times = get_time_pars(time, date)

    where = fp.get_where(times)
    where, times = compute(where, times)
    ndim = len(where)
    if ndim > 0:
        return ndim, times[0], times[-1] + 1
    else:
        return ndim, None, None


###############################################################################
@delayed(pure=True)
def read_lines(inpath):
    with open(inpath, 'r') as inh:
        lines = inh.read().split('\n\n')
        lines = (line.rstrip() for line in lines)
        lines = (line for line in lines if line)
        lines = [line.strip('\r\n') for line in lines]
    return lines


@delayed(pure=True)
def get_time_pars(time, date):
    times = [dt.datetime.strptime(a+' '+b, '%d.%m.%Y %H:%M:%S')
             for a, b in zip(date, time)]
    return np.asarray(times).astype('<M8[us]')


@delayed(pure=True)
def readfunc_brazil(rows, dim, header, dtype, size, where=None):
    idat = []
    for i, row in enumerate(rows):
        if where is not None and i not in where:
            continue

        cols = row.split('\n')
        cols = [col.strip('').split(':', maxsplit=1) for col in cols]
        for col in cols:
            if int(col[0]) == header:
                if len(dim) == 3:
                    classes = col[1].split(';')[:-1]
                    if len(classes) != size:
                        idat += [[ph.common.get_fill(dtype)] * size]
                    else:
                        idat += [classes]
                else:
                    if col[1] == '' or col[1] == 'na':
                        idat += [ph.common.get_fill(dtype)]
                    else:
                        idat += [col[1]]
    return idat


if __name__ == "__main__":
    ph.common.standardlogger()
    path = "/media/tcvanleth/Data/Data2/esp_nd"
    times = ['2011-11-01', '2012-01-01', '2012-03-01']
    for i in range(len(times[:-1])):
        fetch_disdro(path, begin=times[i], end=times[i+1])
