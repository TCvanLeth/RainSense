# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 09:56:00 2015

@author: T.C. van Leth
"""

import datetime as dt
import logging
import os

import numpy as np
import phad as ph
import yaml

from mwlink import inout as io


def fetch_pluvio(indir):
    """
    """
    # fetch parameters
    path = os.path.join(os.path.dirname(__file__), 'pluviometer.yaml')
    with open(path, mode='r') as inh:
        meta = yaml.safe_load(inh)

    # search for files in input directory
    files = os.listdir(indir)
    if len(files) == 0:
        return

    prefix = meta['stations']['forum']['prefix']
    infiles = sorted([x for x in files if prefix in x])

    # read and convert ASCII files
    logging.info('converting pluviometer files to hdf5 format')
    for inname in infiles:
        inpath = os.path.join(indir, inname)

        logging.info('reading %s' % (inname))
        data = read_pluvio(inpath, meta['stations']['forum'])

        if len(data['time']) > 0:
            logging.info('converting %s' % (inname))
            time = ph.TimeScale(data['time'])
            time.step = np.timedelta64(30, 's')
            data = ph.convert_to_set(data, [time], meta)

        logging.info('writing %s to disk' % (inname))
        io.export_ds(data)
    logging.info('finished converting pluviometer files to hdf5 format')


def read_pluvio(inpath, meta):
    # initialize
    tips = []

    # open file and read lines
    with open(inpath) as inh:
        for line in inh:
            line = line.strip()
            if line == '':
                continue
            itime = dt.datetime.strptime(line, "%d/%m/%y %H:%M:%S:%f")
            tips += [itime]

    # put tipping points in array
    tips = np.asarray(tips)

    # create regular time series
    step = ph.common.str_to_td64(meta['attrs']['temporal_resolution'])
    sday = dt.datetime.combine(tips[0].date(), dt.time.min)
    sday = np.datetime64(sday, 'us')
    eday = dt.datetime.combine(tips[-1].date(), dt.time.min)
    eday = np.datetime64(eday, 'us')

    time = np.arange(sday-step, eday, step)
    tips = tips.astype('<M8[us]')

    # create cumulative timeseries
    V = 0.1
    cumrain = V * np.arange(len(tips))

    # convert to regular frequency rain intensity
    ind = np.searchsorted(tips, time)
    cumrain_int = cumrain[ind - 1] + (time - tips[ind - 1]) * (cumrain[ind] - cumrain[ind - 1]) / (tips[ind] - tips[ind - 1])
    R = (cumrain_int[1:] - cumrain_int[:-1]) / step.astype('<m8[s]').astype(int) * 3600

    data = {'time': time[1:],
            'forumchannel_1precip': R,
            'forumchannel_1cumrain': cumrain_int[1:]}
    return data


if __name__ == '__main__':
    path = "/home/tcvanleth/Data/WURex14/Unprocessed_data"
    ph.common.standardlogger()
    fetch_pluvio(path)
