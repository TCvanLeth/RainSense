# -*- coding: utf-8 -*-
"""
Created on Thu Feb 05 17:20:16 2015

@author: T.C. van Leth
"""

import datetime as dt
from ftplib import FTP
import logging
import os
import sys
import urllib
import urllib2

from mwlink import inout as io


def download_knmi(setID, maxtry=10):
    """
    download the latest link data from the KNMI data repository

    arguments:

        outdir -- path of the destination directory

        indir -- path of the backup data directory

    returns:

        list of downloaded file names
    """
    outdir = io.getpath('raw', setID)
    budir = io.getpath('raw', setID)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logging.info('connecting to KNMI server')
    try:
        knmi = FTP('ftppro.knmi.nl')
    except:
        logging.error('could not connect to KNMI server! download aborted')
        return
    knmi.login('tmobile', 'RFlinks4rain')
    knmi.cwd("wageningen")

    logging.info('starting KNMI file download')
    try:
        files = knmi.nlst()
    except:
        sys.stdout.write('\nconnection failure! download aborted.\n')
        return
    i = 0
    n = len(files)
    for inname in files:
        i += 1
        outpath = os.path.join(outdir, inname)
        newpath = os.path.join(budir, inname)
        logging.info('downloading: %s of %s %s' % (i, n, inname))
        if os.path.exists(outpath) or os.path.exists(newpath):
            logging.info('file already exists. skipped')
            continue

        i = 0
        while i != maxtry:
            try:
                with open(outpath, 'w') as outh:
                    knmi.retrbinary("RETR " + inname, outh.write)
                break
            except:
                logging.warning('failed to connect to KNMI! retrying...%s'
                                % (i))
                i += 1
        if i == maxtry:
            logging.error('KNMI file download failed!')
        else:
            logging.info('%s downloaded succesfully', (inname))
    knmi.quit()
    logging.info('all KNMI downloads finished')


def download_swiss(setID, maxtry=10):
    outdir = io.getpath('raw', setID)
    budir = io.getpath('backup', setID)

    locs = {'60': 'NVWA',
            '61': 'Biotechnion',
            '62': 'Bongerd',
            '63': 'Forum2'}

    logging.info('commence downloading EPFL Parsivel files')
    for inum in locs.iterkeys():
        logging.info('downloading from device %s' % (inum))
        # before commencing download, check if most recent file has already
        # been downloaded
        datum = dt.date.today()-dt.timedelta(days=1)
        year = str(datum.year)
        month = str(datum.month).zfill(2)
        day = str(datum.day).zfill(2)

        fname = ('EPFL_Parsivel_data_'+locs[inum] +
                 '_'+year+'_'+month+'_'+day+'.dat')
        outpath = os.path.join(outdir, fname)
        bupath = os.path.join(budir, fname)
        if os.path.exists(outpath) or os.path.exists(bupath):
            logging.info('most recent file from Parsivel %s already exists.'
                         'skipped' % (inum))
            continue

        # commence download and store file in internal memory
        url = "http://lte-d.ch/data/file"+inum+".php"
        try:
            response = urllib2.urlopen(url)
        except urllib2.URLError(e):
            logging.error(e)
            continue
        i = 0
        while i != maxtry:
            try:
                data = response.read()
                break
            except urllib2.httplib.IncompleteRead(e):
                logging.warning('download error: %s! retrying...%s' % (e, i))
                i += 1
        else:
            logging.error('Parsivel %s file download failed!' % (inum))
            continue

        data = data.splitlines()

        # separate records by date
        for dday in xrange(1, 5):
            datum = dt.date.today()-dt.timedelta(days=dday)
            year = str(datum.year)
            month = str(datum.month).zfill(2)
            day = str(datum.day).zfill(2)

            # if a file with these records already exists don't download.
            # it is important not to overwrite as earlier records may become
            # less complete at later dates due to limited server retention!
            fname = ('EPFL_Parsivel_data_'+locs[inum] +
                     '_'+year+'_'+month+'_'+day+'.dat')
            outpath = os.path.join(outdir, fname)
            bupath = os.path.join(budir, fname)
            logging.info('writing %s' % (fname))
            if os.path.exists(outpath) or os.path.exists(bupath):
                logging.warning('file %s already exists! skipped' % (fname))
                continue

            # write ASCII lines to file
            with open(outpath, 'w') as outh:
                for line in data:
                    col = line.split(',')[3].strip('"')
                    try:
                        itime = dt.datetime.strptime(col, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        logging.warning('unreadable record in Parsivel %s'
                                        % (inum))
                        continue
                    if itime.date() == datum:
                        outh.write(line+'\n')
            if os.path.getsize(outpath) == 0:
                os.remove(outpath)
        logging.info('EPFL Parsivel downloads finished')


def download_Veenkampen(setID):
    logging.info('commence downloading Veenkampen files')
    outdir = io.getpath('raw', setID)
    budir = io.getpath('backup', setID)
    outdir = os.path.join(outdir, 'Veenkampen')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    startday = dt.date(2014, 8, 20)
    endday = dt.date.today()
    delta = dt.timedelta(days=1)

    n = (startday-endday).days

    i = 0
    thisday = startday
    while thisday < endday:
        i += 1
        year = str(thisday.year)
        month = str(thisday.month).zfill(2)
        day = str(thisday.day).zfill(2)
        thisday += delta

        inname = 'C_'+year+month+day+'.txt'
        outname = 'Veenkampen_'+inname
        url = ('http://www.met.wau.nl/veenkampen/data/' +
               year+'/'+month+'/'+inname)
        outpath = os.path.join(outdir, outname)
        bupath = os.path.join(budir, 'Veenkampen', outname)
        logging.debug('downloading: %s of %s %s' % (i, n, inname))
        if os.path.exists(outpath) or os.path.exists(bupath):
            logging.debug('file %s already exists! skipped' % (inname))
            continue

        outh = urllib.URLopener()
        try:
            outh.retrieve(url, outpath)
        except IOError(e):
            logging.debug('failed to write file %s: %s' % (outname, e))
    logging.info('Veenkampen downloads finished')
