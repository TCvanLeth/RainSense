# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:21:06 2015

@author: T.C. van Leth
"""

import datetime as dt
import gzip
import logging
import os

import numpy as np
import phad as ph

from mwlink import inout as io


def fetch_Tlink(conf, dirs):
    """
    """
    outdir = dirs.proc_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outname = conf['output']
    outpath = os.path.join(outdir, outname)

    budir = dirs.bu_dir
    if not os.path.exists(budir):
        os.makedirs(budir)

    # select correct files
    indir = dirs.dl_dir
    infiles = sorted(os.listdir(indir))
    if len(infiles) == 0:
        return
    infiles = [name for name in infiles if conf['prefix'] in name]

    # determine filenames
    logging.info('converting link ASCII files to CMLh5 format')
    for i, inname in enumerate(infiles):
        logging.info('reading %s' % (inname))
        inpath = os.path.join(indir, inname)
        data = linkread(inpath, conf['columns'])
        if len(data['time']) > 0:
            logging.info('converting %s' % (inname))
            data = ph.convert_to_set(sortlinks(data, conf))
            logging.info('writing %s' % (inname))
            io.export_ds(data, outpath, daily=conf['daily'])

        # move file to backup directory
        bupath = os.path.join(budir, inname)
        os.rename(inpath, bupath)
    logging.info('finished converting link ASCII files to CMLh5 format')


def linkread(inpath, vmeta):
    """
    Read ASCII file from Tmobile.
    """

    # initialize data containers
    data = {}
    indices = {}
    data['time'] = []
    for name in vmeta.iterkeys():
        data[name] = []
        indices[name] = np.nan

    # load file
    if os.path.splitext(inpath)[1] == '.gz':
        with gzip.open(inpath, 'r') as inh:
            lines = inh.readlines()
    else:
        with open(inpath, 'r') as inh:
            lines = inh.readlines()

    # parse lines
    for i, line in enumerate(lines):
        row = line.strip('\r\n')
        row = row.split(',')
        if i == 0:
            for name in vmeta.iterkeys():
                header = vmeta[name]['header']
                if header in row:
                    indices[name] = row.index(header)

        if i >= 1:
            # parse timecode
            itime = dt.datetime.strptime(row[8], "%Y%m%d%H%M%S")
            itime = np.datetime64(itime.replace(tzinfo=ph.common.UTC()), 'ns')

            data['time'] += [itime]

            # parse other data
            for name in vmeta.iterkeys():
                index = indices[name]
                if np.isnan(index):
                    data[name] += [np.nan]
                    continue

                col = row[index]

                # read entries
                if col == '':
                    data[name] += [np.nan]
                else:
                    data[name] += [col]

    # convert lists to numpy arrays
    for name in data.iterkeys():
        data[name] = np.asarray(data[name])
        if data[name].dtype.char == 'S':
            data[name] = np.char.strip(data[name])

        # convert to appropriate datatype
        if name == 'time':
            continue
        elif vmeta[name]['dtype'] == 'int':
            data[name] = data[name].astype(float).astype(int)
        else:
            data[name] = data[name].astype(vmeta[name]['dtype'])

        # convert decimal arcseconds to decimal degrees
        if name in ['near_lat', 'near_lon', 'near_alt',
                    'far_lat', 'far_lon', 'far_alt']:
            data[name] = data[name]/3600.
    return data


def sortlinks(data, conf):
    """
    """
    tstep = '15t'
    ATPC = False

    # split by link
    data = split(data, 'linkID')

    # split by link channel
    for l_ID, ilink in data.iteritems():
        data[l_ID] = split(ilink, 'channelID')

    # process link metadata
    meta = {'links': {}}
    meta['attrs'] = {'data_set_id': conf['data_set_id'],
                     'file_format': 'CMLh5',
                     'file_format_version': '0.1'}

    for l_ID, ilink in data.iteritems():
        meta['links'][l_ID] = {'attrs': {}, 'channels': {}}

        n_sID = data[l_ID]['near_sID'][0]
        n_lat = data[l_ID]['near_lat'][0]
        n_lon = data[l_ID]['near_lon'][0]
        if 'near_alt' in data[l_ID]:
            n_alt = data[l_ID]['near_alt'][0]
        else:
            n_alt = np.nan

        f_sID = data[l_ID]['far_sID'][0]
        f_lat = data[l_ID]['far_lat'][0]
        f_lon = data[l_ID]['far_lon'][0]
        if 'far_alt' in data['l_ID']:
            f_alt = data[l_ID]['far_alt'][0]
        else:
            f_alt = np.nan

        meta['links'][l_ID]['attrs'] = {'site_a_id': n_sID,
                                        'site_a_latitude': n_lat,
                                        'site_a_longitude': n_lon,
                                        'site_a_altitude': n_alt,
                                        'site_b_id': f_sID,
                                        'site_b_latitude': f_lat,
                                        'site_b_longitude': f_lon,
                                        'site_b_altitude': f_alt,
                                        'system_manufacturer': conf['d_ID']}

    # remove sloppily logged link
    if '3024A-9015A-1' in data:
        data.pop('3024A-9015A-1')

    # split duplex channels
    for l_ID, ilink in data.iteritems():
        link2 = {}
        for c_ID, chan in ilink.iteritems():
            chan = split(chan, 'near_sID')
            for i, site in enumerate(chan.itervalues()):
                if len(chan.keys()) == 2:
                    c_ID2 = c_ID+'#'+str(i+1)
                    link2[c_ID2] = site
                else:
                    link2[c_ID] = site
        data[l_ID] = link2

    # process channel metadata
    for l_ID, ilink in data.iteritems():
        for c_ID, ichan in ilink.iteritems():
            nID = meta['links'][l_ID]['attrs']['site_a_id']
            if ichan['near_sID'][0] == nID:
                side = 'near'
            else:
                side = 'far'
            attrs = {'frequency': ichan['frequency'][0],
                     'polarization': 'horizontal',
                     'receiver_side': side,
                     'ATPC': ATPC,
                     'temporal_resolution': tstep}
            meta['links'][l_ID]['channels'][c_ID]['attrs'] = attrs

            # remove duplicate metadata
            remlist = ['channelID', 'linkID', 'frequency', 'period',
                       'near_lat', 'near_lon', 'near_alt', 'near_sID',
                       'far_lat', 'far_lon', 'far_alt', 'far_sID']
            for item in remlist:
                if item in ichan:
                    ichan.pop(item)

            for v_ID in ichan.iterkeys():
                vmet = conf['columns'][v_ID]
                meta['links'][l_ID]['channels'][c_ID]['variables'][v_ID] = vmet
    return data, meta


###############################################################################
# convenience functions

def split(data, sorter):
    """
    sort and split link data.

    takes a dictionary of lists. sorts the dictionary by the values of the
    sorter list and splits all lists for nonequal values of sorter.

    parameters:
        data: dict of lists

        sorter: string. keyword to indexer in data

    returns:
        grdata: dict of lists. sorted and splitted dataset
    """
    index = np.argsort(data[sorter])
    for name in data.keys():
        data[name] = data[name][index]
    cond = np.where(data[sorter][1:] != data[sorter][:-1])[0]+1
    for name in data.keys():
        data[name] = np.split(data[name], cond)

    grdata = {}
    for i, isort in enumerate(data[sorter]):
        grdata[isort[0]] = {}
        for name in data.keys():
            grdata[isort[0]][name] = data[name][i]
    return grdata
