# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:04:39 2015

@author: T.C. van Leth
"""

import datetime as dt
import logging
import os

import numpy as np
import phad as ph
import yaml

from mwlink import inout as io


def fetch_veen(indir, begin=None, end=None):
    """
    """
    # fetch parameters
    path = os.path.join(os.path.dirname(__file__), 'veenkampen.yaml')
    with open(path, mode='r') as inh:
        meta = yaml.safe_load(inh)

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

    # read ASCII files
    logging.info('converting veenkampen files to hdf5 format')
    mdat = []
    for date in dates:
        dfiles = [x for x in files if date in x]
        if not dfiles:
            continue
        prefix = meta['prefix']
        infiles = sorted([x for x in dfiles if prefix in x])
        if not infiles:
            continue

        for inpath in infiles:
            logging.info('reading %s' % (inpath))
            data = veenread(inpath, meta)
            start = data['time'].astype('<M8[D]')[0]
            stop = data['time'].astype('<M8[D]')[-1] + 1
            chunks = len(data['time']) * 2
            time = ph.TimeScale(data['time'], chunksize=chunks,
                                start=start, stop=stop)
            time.step = None

            step = meta['temporal_resolution']
            data = ph.convert_to_set(data, {'time':time}, meta, chunksize=chunks)
            data = ph.regularize(data, [time], [step])
            mdat.append(data)

    logging.info('merging data')
    mdat = ph.Network.merged(mdat)
    io.export_ds(mdat)
    logging.info('finished converting veenkampen files to hdf5 format')


def veenread(inpath, meta):
    # initialize lists
    data = {}
    dtypes = {}
    indices = {}
    times = []
    for sID, smet in meta['stations'].items():
        for cID, cmet in smet['channels'].items():
            for vID, vmet in cmet['variables'].items():
                name = sID+cID+vID
                indices[name] = vmet['header']
                data[name] = []
                dtypes[name] = vmet['dtype']
    if not len(data):
        return

    # load file
    with open(inpath, 'r') as inh:
        lines = (line.rstrip() for line in inh)
        lines = (line for line in lines if line)
        rows = [line.strip('\r\n').split(',') for line in lines]
    # parse lines
    ltime = np.datetime64('1970-01-01', 'us')
    for row in rows:
        # parse timecode
        row[0] = row[0].strip('"')
        itime = np.datetime64(' '.join((row[0], row[1])), 'us')

        # skip duplicate records
        if itime <= ltime:
            logging.warning('duplicate records in %s' % (inpath))
            continue
        times += [itime]

        # parse other data
        for name in data.keys():
            index = indices[name]
            col = row[index]
            if col in ['-999.', '-999', '-999.0000', '']:
                data[name] += [np.nan]
            elif (index == 19 and float(col) < 0):
                data[name] += [np.nan]
            else:
                data[name] += [col]
        ltime = itime
    data['time'] = np.asarray(times)

    # convert lists to numpy arrays
    for name in indices.keys():
        idat = data[name]
        dtype = dtypes[name]

        idat = np.asarray(idat)
        if idat.dtype.kind in ('S', 'U'):
            idat = np.where(idat == '', '0', idat)
            idat = np.char.strip(idat)

        # convert to appropriate datatype
        if dtype == 'int':
            idat = idat.astype(float).astype(int)
        else:
            idat = idat.astype(dtype)
        data[name] = idat
    return data


if __name__ == "__main__":
    ph.common.standardlogger()
    path = "/home/tcvanleth/Data/WURex14/Unprocessed_data"
    fetch_veen(path)