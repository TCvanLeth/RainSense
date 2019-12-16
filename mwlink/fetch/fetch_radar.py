# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 08:58:48 2015

@author: T.C. van Leth
"""

import datetime as dt
import os
import sys

from netCDF4 import Dataset
import numpy as np
import xray

from mwlink import inout as io


class Radar_fetcher:
    '''
    Fetch radar data from ASCII files and merge into HDF5 files.

    arguments:

    - indir  --> directory where radar files are located

    - outdir --> directory where HDF5 files are located

    - budir  --> directory where radar files should be copied
                 after processing.
    '''
    def __init__(self, setID):
        self.indir = os.path.join(io.getpath('raw', setID), 'Radar')
        self.budir = os.path.join(io.getpath('backup', setID), 'Radar')

        self.outpath = io.getpath('radar_l1', setID)
        self.outdir = os.path.dirname(self.outpath)
        self.outpath = self.outpath+'.h5'

        self.geo = {'pixel1': {'latitude': 51.971,
                               'longitude': 5.677,
                               'altitude': np.nan},
                    'pixel2': {'latitude': 51.980,
                               'longitude': 5.678,
                               'altitude': np.nan},
                    'pixel3': {'latitude': 51.980,
                               'longitude': 5.664,
                               'altitude': np.nan},
                    'pixel4': {'latitude': 51.971,
                               'longitude': 5.665,
                               'altitude': np.nan}}

    def fetch(self):
        """
        fetch the radar data.
        """
        sys.stdout.write('\nprocessing radar files...\n')
        # fetch linkpath averaged gauge adjusted radar data
        self.prefix = 'radar_unadjusted_'
        self.prefix2 = 'radar_gaugeadjusted_'
        self.convert_path()

        # fetch gauge adjusted radar pixel data
        self.prefix = 'Radar_5min_RAC_MFBS_Wageningen'
        self.prefix2 = 'Radar_5min_RAP_Wageningen'
        self.convert_pixel()

    def convert_pixel(self):
        """
        Fetch radar pixel data from file

        arguments:

        - dname --> name of the dataset, included in the filename (string)
        """
        # iterate over each radar pixel
        for inum in range(1, 5):
            # construct hdf group name
            grp = 'radar_pixel_'+str(inum)
            geo = self.geo['pixel'+str(inum)]

            # construct input filename
            unname = self.prefix+str(inum)+'.dat'
            adname = self.prefix2+str(inum)+'.dat'
            self.unpath = os.path.join(self.indir, unname)
            self.adpath = os.path.join(self.indir, adname)
            if not os.path.exists(self.unpath):
                continue
            sys.stdout.write('\nprocessing %s of 4 %s\n' % (inum, unname))

            # construct backup filename
            bupath = os.path.join(self.budir, unname)
            bupath2 = os.path.join(self.budir, adname)

            # initialize data container
            data = {}

            # open unadjusted file
            i = 0
            with open(self.unpath, 'r') as inh:
                for line in inh:
                    if i > 3:  # start of columns

                        # read contents of file
                        line = line.strip('\n\r')
                        row = line.split(',')
                        itime = dt.datetime.strptime(row[0], '%Y%m%d%H%M')
                        datum = itime.date()

                        # initialize lists
                        if datum not in data:
                            data[datum] = {}
                            data[datum]['datetime'] = []
                            data[datum]['radar_rain_unadjusted'] = []
                            data[datum]['radar_rain_adjusted'] = []

                        # parse data
                        data[datum]['datetime'] += [itime]
                        if row[1] == '655.35':  # error value
                            data[datum]['radar_rain_unadjusted'] += [np.nan]
                        else:
                            data[datum]['radar_rain_unadjusted'] += [row[1]]
                    i += 1

            # open adjusted file
            i = 0
            with open(self.adpath, 'r') as inh:
                for line in inh:
                    if i > 3:  # start of columns

                        # read contents of file
                        line = line.strip('\n\r')
                        row = line.split(',')
                        itime = dt.datetime.strptime(row[0], '%Y%m%d%H%M')
                        datum = itime.date()

                        # parse data
                        if row[1] == '655.35':  # error value
                            data[datum]['radar_rain_adjusted'] += [np.nan]
                        else:
                            data[datum]['radar_rain_adjusted'] += [row[1]]
                    i += 1

            # write data to netCDF file
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            for day in data.iterkeys():
                sys.stdout.write('\nprocessing radar %s\n' % day)
                self._pixwrite(self.outpath, data[day], grp, geo)

            # place original file in backup folder
            os.rename(self.unpath, bupath)
            os.rename(self.adpath, bupath2)

    def _pixwrite(self, outpath, data, grp, geo):
        """
        """
        # convert to numpy array
        signal = {}
        datum = np.asarray(data['datetime'], dtype='datetime64[us]')
        signal['radar_rain_unadjusted'] = np.asarray(data['radar_rain_unadjusted'], dtype='f')
        signal['radar_rain_adjusted'] = np.asarray(data['radar_rain_adjusted'], dtype='f')

        # create xray dimensions
        time = xray.Coordinate('time', datum)
        loc = xray.Coordinate('location', [grp])

        # create xray secondary coordinates
        lat = xray.DataArray([geo['latitude']], coords=[loc],
                             attrs={'unit': 'deg'})
        lon = xray.DataArray([geo['longitude']], coords=[loc],
                             attrs={'unit': 'deg'})
        h = xray.DataArray([geo['altitude']], coords=[loc],
                           attrs={'unit': 'm'})

        # create xray DataArrays
        for name in signal.iterkeys():
            signal[name] = xray.DataArray(signal[name], coords=[time],
                                          attrs={'unit': 'mm/5min'})

        # create xray Dataset
        xsign = xray.Dataset(variables=signal,
                             coords=dict([('location', loc),
                                          ('latitude', lat),
                                          ('longitude', lon),
                                          ('altitude', h)]))
        # write to netCDF4 file
        if os.path.exists(outpath):
            # clunky code to work around the fact that we cannot write
            # incrementally.
            with Dataset(outpath) as outh:
                diskgrps = outh.groups
            xback = {}
            for igrp in diskgrps:
                with xray.open_dataset(outpath, group='/'+igrp) as grph:
                    dset = grph.load_data()
                    xback.update({igrp: dset})
            if grp in diskgrps:
                xold = xback[grp]
                xsign = xray.concat((xold, xsign), dim='time')
            xback[grp] = xsign
            os.remove(outpath)
            i = 0
            for igrp in xback.iterkeys():
                if i == 0:
                    xback[igrp].to_netcdf(outpath, mode='w', group='/'+igrp)
                else:
                    xback[igrp].to_netcdf(outpath, mode='a', group='/'+igrp)
                i += 1
        else:
            pass

    def _pathwrite(self, outpath, data):
        """
        """
        grp = 'linkpath'
        # convert to numpy array
        signal = {}
        datum = np.asarray(data['datetime'], dtype='datetime64[us]')
        signal['radar_rain_unadjusted'] = np.asarray(data['radar_rain_unadjusted'], dtype='f')
        signal['radar_rain_adjusted'] = np.asarray(data['radar_rain_adjusted'], dtype='f')

        # create xray dimensions
        time = xray.Coordinate('time', datum)

        # create xray DataArrays
        for name in signal.iterkeys():
            signal[name] = xray.DataArray(signal[name], coords=[time],
                                          attrs={'unit': 'mm/5min'})

        # create xray Dataset
        xsign = xray.Dataset(variables=signal)

        # write to netCDF4 file
        if os.path.exists(outpath):
            # clunky code to work around the fact that we cannot write
            # incrementally.
            with Dataset(outpath) as outh:
                diskgrps = outh.groups
            xback = {}
            for igrp in diskgrps:
                with xray.open_dataset(outpath, group='/'+igrp) as grph:
                    dset = grph.load_data()
                    xback.update({igrp: dset})
            if grp in diskgrps:
                xold = xback[grp]
                xsign = xray.concat((xold, xsign), dim='time')
            xback[grp] = xsign
            os.remove(outpath)
            i = 0
            for igrp in xback.iterkeys():
                if i == 0:
                    xback[igrp].to_netcdf(outpath, mode='w', group='/'+igrp)
                else:
                    xback[igrp].to_netcdf(outpath, mode='a', group='/'+igrp)
                i += 1
        else:
            xsign.to_netcdf(outpath, mode='w', group='/'+grp)

    def convert_path(self):
        """
        Fetch the pathintegrated radar average data from file

        Arguments:

        - dname --> name of the dataset, included in the filename (string)
        """
        grp = 'linkpath'

        # search for files in directory
        undir = self.prefix[:-1]
        undir = os.path.join(self.indir, undir)
        unfiles = sorted(os.listdir(undir))
        n = len(unfiles)
        if n == 0:
            return

        j = 0
        for unname in unfiles:
            j += 1
            addir = os.path.join(self.indir, self.prefix2[:-1])
            adname = self.prefix2+unname[len(self.prefix):]
            self.unpath = os.path.join(undir, unname)
            self.adpath = os.path.join(addir, adname)
            sys.stdout.write('\nprocessing %s of %s %s\n' % (j, n, unname))

            # analyse input path
            datum, bupath = self._path_dissect(unname, self.prefix)
            datum, bupath2 = self._path_dissect(adname, self.prefix2)

            if os.path.exists(bupath):
                os.remove(self.unpath)
                os.remove(self.adpath)
                continue

            # initialize lists
            data = {}
            data['datetime'] = []
            data['radar_rain_unadjusted'] = []
            data['radar_rain_adjusted'] = []

            # read unadjusted file
            i = 0
            with open(self.unpath, 'r') as inh:
                for line in inh:
                    if i > 1:  # start of columns

                        # parse contents of file
                        line = line.strip('\n\r')
                        row = line.split()
                        itime = dt.datetime.strptime(row[0], '%Y%m%d%H%M')
                        data['datetime'] += [itime]
                        if row[1] == '-999.0000':  # error value
                            data['radar_rain_unadjusted'] += [np.nan]
                        else:
                            data['radar_rain_unadjusted'] += [row[1]]
                    i += 1

            # read adjusted file
            i = 0
            with open(self.adpath, 'r') as inh:
                for line in inh:
                    if i > 1:  # start of columns

                        # parse contents of file
                        line = line.strip('\n\r')
                        row = line.split()
                        itime = dt.datetime.strptime(row[0], '%Y%m%d%H%M')
                        if row[1] == '-999':  # error value
                            data['radar_rain_adjusted'] += [np.nan]
                        else:
                            data['radar_rain_adjusted'] += [row[1]]
                    i += 1

            # write data to hdf5 file
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            self._pathwrite(self.outpath, data)

            # place original file in backup folder
            os.rename(self.unpath, bupath)
            os.rename(self.adpath, bupath2)

    def _path_dissect(self, infile, prefix):
        """
        Extract date and backup pathname from input filename
        """
        # extract date from filename
        date = os.path.splitext(infile)[0][len(prefix):]
        datum = dt.datetime.strptime(date, '%Y%m%d').date()

        # create backup directory
        budir = os.path.join(self.budir, prefix[:-1])
        bupath = os.path.join(budir, infile)
        if not os.path.exists(budir):
            os.makedirs(budir)

        return datum, bupath

    def _path_construct(self, datum):
        """
        construct the appropriate pathname of the output HDF5 file

        Arguments:

        - datum --> datetime object containing the date of the exported records
        """
        # construct output path
        year = str(datum.year)
        month = str(datum.month).zfill(2)
        day = str(datum.day).zfill(2)
        outdir2 = os.path.join(self.outdir, year+'_'+month)
        self.outpath = os.path.join(outdir2, 'WURex14_'+year+'_'+month+'_'+day+'.h5')

        # create output directory
        if not os.path.exists(outdir2):
            os.mkdir(outdir2)
