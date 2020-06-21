# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:51:50 2015

@author: T. C. vanleth
"""
import os, logging

import pyhad as ph
from pyhad import common
from pyhad import inout_common as io

from rainsense import settings


def export_ds(data, daily=False, level=None, pro_id=None, merge=False):
    """
    write data container to hdf5 file
    """
    logging.info('writing data to database')

    if level is not None:
        data.attrs['level'] = level
    if pro_id is not None:
        data.attrs['pro_id'] = pro_id
    level = data.attrs['level']
    setID = data.name
    proID = data.attrs['pro_id']

    outpath = getpath(level, setID, proID=proID)
    if daily:
        datlist = data.groupby('time.date')
        for dID, idat in datlist:
            subpath = makename(setID, dID, outpath)
            idat.store(subpath)
    elif merge:
        if os.path.exists(outpath + '.h5'):
            cls = type(data)
            data2 = io.from_hdf(cls, outpath + '.h5')
            data = cls.merged([data2, data])
        data.store(outpath)
    else:
        i = 0
        outpath2 = outpath
        while os.path.exists(outpath2 + '.h5') or os.path.exists(outpath2 + '.h5.temp'):
            i += 1
            outpath2 = outpath + '_' + str(i)
        data.store(outpath2)


def import_ds(level, setID, pro_id=None, times=None, conform=False, **kwargs):
    """
    retrieve data container from hdf5 files
    """
    cls = get_class(level)
    inpath = getpath(level, setID, proID=pro_id)
    if os.path.isdir(inpath):
        days = io.checktimes(times)
        data = []
        for thisday in days:
            subpath = makename(setID, thisday, inpath)
            try:
                data += [io.from_hdf(cls, subpath, **kwargs)]
            except IOError:
                continue

        if len(data) > 0:
            data = ph.merge(data).rechunk(common.CHUNKSIZE, affect_aux=True)
        else:
            data = cls()
            data.attrs['level'] = level
            data.name = setID
            data.attrs['pro_id'] = pro_id
    else:
        if level in ('powerlaw', 'calibration'):
            infunc = io.from_yaml
        else:
            infunc = io.from_hdf
        try:
            data = infunc(cls, inpath, **kwargs)
        except IOError:
            data = cls()
            data.attrs['level'] = level
            data.name = setID
            data.attrs['pro_id'] = pro_id

    if times is not None:
        data = data.sel(time=times)
    if conform:
        data = data.conform()
    return data


###############################################################################
# path manipulation
def get_class(level):
    if level == 'scatter' or level == 'dsd2' or level == 'conditions':
        return ph.Channel
    else:
        return ph.Network


def getpath(level, setID, proID=None):
    if level == 'raw':
        path = os.path.join(settings.datapath, 'Ingest')
    elif level == 'backup':
        path = os.path.join(settings.datapath, setID, 'raw')
    elif level == 'calibration':
        path = os.path.join(settings.datapath, 'calibration', setID)
    elif proID == 'NA' or proID is None:
        name = '_'.join((setID, level))
        path = os.path.join(settings.datapath, setID, name)
    else:
        name = '_'.join((setID, level, proID))
        path = os.path.join(settings.datapath, setID, level, name)
    return path


def makename(setID, datum, path):
    datum = str(datum.astype('<M8[D]')).replace('-', '_')
    fname = setID + '_' + os.path.basename(path) + '_' + datum
    return os.path.join(path, datum[:-3], fname)