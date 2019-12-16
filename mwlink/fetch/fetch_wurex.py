# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 11:03:44 2015

@author: T.C. van Leth
"""

import datetime as dt
import logging
import os

from dask import delayed, compute
import dask.array as da
import numpy as np
import phad as ph
from tqdm import tqdm
import yaml

from mwlink import inout as io
from mwlink.fetch import fetch_pars_common as fp


def fetch_link(indir, begin=None, end=None):
    """
    read the plain ASCII files produced by the WURex14 loggers and convert
    to extended CMLH5 format.
    """
    # fetch parameters
    metapath = os.path.join(os.path.dirname(__file__), 'wurex.yaml')
    with open(metapath, mode='r') as inh:
        meta = yaml.safe_load(inh)

    sdiffs = ((dt.datetime(2015, 6, 5, 7, 19, tzinfo=ph.common.UTC()) -
               dt.datetime(2015, 3, 1, 23, 27, tzinfo=ph.common.UTC())),
              (dt.datetime(2015, 7, 3, 8, 3, 30, tzinfo=ph.common.UTC()) -
               dt.datetime(2015, 7, 3, 8, 2, 43, tzinfo=ph.common.UTC())))
    spoints = (dt.datetime(2015, 6, 5, 7, 24, 30, tzinfo=ph.common.UTC()),
               dt.datetime(2015, 7, 3, 8, 4, tzinfo=ph.common.UTC()))

    # define experiment period
    if begin is None:
        begin = '2014-08-20'
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

    logging.info('converting link ASCII files to CMLh5 format')
    prefixes = ['CR1000-0730-FORUM_data_link_signals_',
                'CR1000-0730-FORUM_status_signals_',
                'CR1000_Biotechnion']

    ntime = {}
    start = {}
    stop = {}
    indices = {}
    for prefix in prefixes:
        infiles = [x for x in files if prefix in x]
        if not infiles:
            continue
        dtypes, headers = prepare_dicts(prefix, meta)
        for date in dates:
            dfiles = [x for x in infiles if date in x]
            if not dfiles:
                continue

            for inpath in dfiles:
                if os.stat(inpath).st_size > 0:
                    ntime[inpath], start[inpath], stop[inpath], indices[inpath] = getlength(inpath, headers)

    mdat = []
    for prefix in tqdm(prefixes, desc="data types"):
        infiles = [x for x in files if prefix in x]
        if not infiles:
            continue
        dtypes, headers = prepare_dicts(prefix, meta)
        for date in tqdm(dates, desc='dates'):
            dfiles = [x for x in infiles if date in x]
            if not dfiles:
                continue

            for inpath in dfiles:
                if os.stat(inpath).st_size > 0:
                    data = read(inpath, indices, dtypes)
                    if prefix == 'CR1000_Biotechnion':
                        logging.info('synchronizing logger times')
                        data = time_correct(data, spoints)

                    time = da.from_delayed(data['time'], (ntime[inpath],), '<M8[us]')
                    chunks = len(time) * 2
                    time = ph.TimeScale(time, chunksize=chunks,
                                        start=start[inpath], stop=stop[inpath])
                    time.step = None
                    if prefix == 'CR1000-0730-FORUM_data_link_signals_':
                        step = '50ms'
                    else:
                        step = meta['temporal_resolution']
                    data = ph.convert_to_set(data, {'time': time}, meta,
                                             chunksize=chunks)

                    logging.info('regularizing %s' % (inpath))
                    data = ph.regularize(data, [time], [step])
                    mdat.append(data)
    if mdat:
        logging.info('merging data')
        mdat = ph.merge(mdat, chunksize=100000).rechunk(100000)

        # correct temperatures for mistake in logger processing
        logging.info('correcting temperatures')
        mdat = temp_correct(mdat)

        # add aggregate values
        #mean = mdat.resample(dim='time', how='mean', rename=True)
        #stdv = mdat.resample(dim='time', how='std', rename=True)
        #maxi = mdat.resample(dim='time', how='max', rename=True)
        #mini = mdat.resample(dim='time', how='min', rename=True)
        #mdat = ha.Network.merged([mdat, mean, stdv, maxi, mini])

        io.export_ds(mdat)
    logging.info('finished converting link files')


def prepare_dicts(prefix, meta):
    dtypes = {}
    headers = {}
    for sID, smet in meta['stations'].items():
        for cID, cmet in smet['channels'].items():
            for vID, vmet in cmet['variables'].items():
                if vmet['prefix'] == prefix:
                    name = sID + cID + vID
                    headers[name] = vmet['header']
                    dtypes[name] = vmet['dtype']
    return dtypes, headers


def read(inpath, indices, dtypes):
    """
    Read ASCII file from Forum link datalogger and convert to binary hdf
    format.
    """
    logging.info('reading %s' % (inpath))
    rows = read_lines(inpath)
    times = get_time(rows[4:], 0)

    where = fp.get_where(times)
    data = {}
    data['time'] = times[where]
    for name, index in indices[inpath].items():
        dtype = dtypes[name]
        idat = readfunc(rows[4:], index, dtype)
        data[name] = fp.convert(idat, dtype, ())
    return data


def getlength(inpath, headers):
    logging.info('reading %s' % (inpath))
    rows = read_lines(inpath)
    times = get_time(rows[4:], 0)
    where = fp.get_where(times)
    head, where, times = compute(rows[1], where, times)

    # determine column order
    indices = {}
    for name in headers.keys():
        if headers[name] in head:
            indices[name] = head.index(headers[name])

    ndim = len(where)
    if ndim > 0:
        return ndim, times[0], times[-1] + 1, indices
    else:
        return ndim, None, None, indices


###############################################################################
@delayed(pure=True)
def read_lines(inpath):
    with open(inpath, 'r') as inh:
        lines = (line.rstrip() for line in inh)
        lines = [line for line in lines if line]
        lines = [line.strip('\r\n').split(',') for line in lines]
    return lines


@delayed(pure=True)
def get_time(rows, time_col):
    times = np.asarray([x[time_col] for x in rows], dtype=str)
    return np.char.strip(times, '"').astype('<M8[us]')


@delayed(pure=True)
def readfunc(rows, index, dtype):
    idat = []
    for row in rows:
        col = row[index]
        if col == '"NAN"':
            idat += [ph.common.get_fill(dtype)]
        else:
            idat += [col]
    return idat


###############################################################################
# corrections for specific datalogging mistakes

def temp_correct(data):
    '''
    Correct mistake in datalogger processing.
    Remove when no longer needed!
    '''
    temp38 = data['ral_38', 'auxiliary', 'temp_rx']
    grnd38 = data['ral_38', 'temp', 'ground']
    grnd38 = grnd38.reindex_like(temp38)

    # convert voltages to degrees celcius (see docs from Bradford)
    data['ral_38', 'auxiliary', 'temp_rx'] = (temp38 - grnd38) * 0.1
    del data['ral_38', 'temp']

    if 'ral_26' in data:
        temp26 = data['ral_26', 'auxiliary', 'temp_rx']
        grnd26 = data['ral_26', 'temp', 'ground']
        grnd26 = grnd26.isel(time_offset=0).drop(['time_offset']).reindex_like(temp26)
        data['ral_26', 'auxiliary', 'temp_rx'] = (temp26 - grnd26) * 0.1
        del data['ral_26', 'temp']
    return data


def time_correct(data, spoints, sdiffs=None, T0=25, C1=3.5E-8, close=False):
    """
    adjust timestamps of an unsynchronised logger using manual synchronisation

    spoints --> synchronisation points in time (according to logger)
    sdiffs --> timedifferences with 'true' time at synchronisation points
    T0 --> crystal parabolic temperature dependancy parameter
    C1 --> crystal parabolic temperature dependancy parameter
    """
    # TO-DO: leap second
    temp = '"CR1000temp"'
    step = 30

    n = len(spoints)
    for i in range(n):
        if i == 0:
            cond = data['time'] < spoints[i]
        elif close:
            cond = data['time'] > spoints[i-1]
        else:
            cond = (data['time'] > spoints[i-1]) & (data['time'] < spoints[i])
        told = data['time'][cond]

        # parabolic temperature dependancy
        m = len(told)
        T = np.tril(np.tile(data[temp][cond][::-1], (m, 1)))

        tnew = map(ph.common.dt_to_posix, spoints[i])
        tnew = tnew - step*np.sum(1-C1*(T-T0)**2, axis=1)**-1
        tnew = map(ph.common.roundtime, map(dt.datetime.utcfromtimestamp, tnew),
                   step)
        data['time'][cond] = tnew

        # interpolate to rounded timestamps
        stime = map(ph.common.dt_to_posix, told)
        gtime = map(ph.common.dt_to_posix, data['time'][cond])
        dub = np.where(data['time'][:-1] == data['time'][1:])

        for name in data.keys():
            if data[name].dtype == float:
                data[name][cond] = np.interp(gtime, stime, data[name][cond])
            data[name] = np.delete(data[name], dub)
    return data


if __name__ == "__main__":
    from dask.diagnostics import ProgressBar

    ph.common.standardlogger()
    path = "/home/tcvanleth/Data/WURex14/Unprocessed_data"
    times = np.arange(np.datetime64('2014-08-20'), np.datetime64('2014-10-10'), np.timedelta64(5, 'D'))
    for i in tqdm(range(len(times) - 1), desc="batch"):
        fetch_link(path, begin=times[i], end=times[i+1])

    dat = []
    path = '/media/data/Data2/WURex14/WURex14_link_l1'
    dat.append(ph.inout_common.from_hdf(ph.Network, path, hi_res=True))
    for i in range(1, len(times) - 1):
        path = '/media/data/Data2/WURex14/WURex14_link_l1_'+str(i)
        dat.append(ph.inout_common.from_hdf(ph.Network, path, hi_res=True))

    outpath = '/media/data/Data2/WURex14/WURex14_link_l1_comb'
    print("merging")
    cdat = ph.Network.merged(dat)
    print("writing to disk")
    with ProgressBar():
        cdat.store(outpath)
