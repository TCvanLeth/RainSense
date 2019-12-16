# -*- coding: utf-8 -*-
"""
"""

from dask import delayed
import numpy as np
import phad as ph


def create_bins(device):
    indexes = {}
    if device:
        for dID, idim in device['dimension_scales'].items():
            label = idim['attrs']['quantity']
            indexes[dID] = ph.Variable(idim['bins'], label, name=dID,
                                        attrs=idim['attrs'])
            if label == dID:
                indexes[dID] = indexes[dID].to_index()
                indexes[dID].resolution = None
    return indexes


def prepare_dicts(sID, smet, device):
    n_v = len(device['dimension_scales']['velocity']['bins'])
    n_D = len(device['dimension_scales']['diameter']['bins'])

    dtypes = {}
    dims = {}
    headers = {}
    for cID, cmet in smet['channels'].items():
        for vID, vmet in cmet['variables'].items():
            name = sID+cID+vID
            dtypes[name] = vmet['dtype']
            dims[name] = vmet['dimensions']
            headers[name] = vmet['header']
    return dtypes, dims, headers, {'velocity': n_v, 'diameter':n_D}


@delayed(pure=True)
def get_where(times):
    return np.where(np.insert(times[1:] > times[:-1], 0, True))[0]

    mat = np.greater.outer(times, times)
    return np.where(np.all(~np.tril(~mat, k=-1), axis=1))[0]


@delayed(pure=True)
def convert(idat, dtype, shape):
    idat = np.asarray(idat)
    if idat.dtype.kind in ('S' 'U'):
        idat = np.where(idat == '', '0', idat)
        idat = np.char.strip(idat)

    # convert to appropriate datatype
    if dtype == 'int':
        idat = idat.astype(float).astype(int)
    else:
        idat = idat.astype(dtype)

    # reshape to appropriate dimensions
    if len(shape) == 2:
        idat = idat.reshape((-1, *shape))
    return idat