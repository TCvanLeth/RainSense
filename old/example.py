# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:35:19 2015

@author: T.C. van Leth
"""

import numpy as np
import h5py

filename = 'D:\example.h5'

# create some fake data
n = 1000
dt = 60
Rx = np.random.randn(n)
Tx = np.random.randn(n)
time = np.arange(1000)


# channel attributes
attr1 = {'frequency': 38,
         'polarisation': 'V',
         'ATPC': True,
         'sampling_type': 'mean',
         'TX_site': 'siteA',
         'RX_site': 'siteB',
         'temporal_resolution': dt}

attr2 = {'frequency': 38,
         'polarisation': 'V',
         'ATPC': True,
         'sampling_type': 'mean',
         'TX_site': 'siteB',
         'RX_site': 'siteA',
         'temporal_resolution': dt}

attr3 = {'ID': 'siteA_siteB',
         'manufacturer': 'Nokia',
         'model': 'Flexihopper'}

# create file
with h5py.File(filename, mode='w') as outh:
    # write groups
    cml1 = outh.create_group('cml_1')
    for key in attr3.iterkeys():
        cml1.attrs[key] = attr3[key]

    chan1 = cml1.create_group('channel_1')
    for key in attr1.iterkeys():
        chan1.attrs[key] = attr1[key]

    # write variables
    chan1.create_dataset('Rx', data=Rx)
    chan1['Rx'].attrs['units'] = 'dBm'
    chan1.create_dataset('Tx', data=Tx)
    chan1['Tx'].attrs['units'] = 'dBm'

    # write time dimension
    chan1.create_dataset('time', data=time)
    chan1['time'].attrs['units'] = 'seconds since 2015-01-01 UTC'
    chan1['time'].attrs['calendar'] = 'proleptic_gregorian'
    chan1['Rx'].dims.create_scale(chan1['time'], 'time')
    chan1['Rx'].dims[0].attach_scale(chan1['time'])
    chan1['Tx'].dims[0].attach_scale(chan1['time'])

    chan2 = cml1.create_group('channel_2')
    for key in attr2.iterkeys():
        chan2.attrs[key] = attr2[key]

    chan2.create_dataset('Rx', data=Rx)
    chan2['Rx'].attrs['units'] = 'dBm'
    chan2.create_dataset('Tx', data=Tx)
    chan2['Tx'].attrs['units'] = 'dBm'

    chan2['time'] = chan1['time']
#    chan2['time'].attrs['units'] = 'seconds since 2015-01-01'
#    chan2['Rx'].dims.create_scale(chan2['time'], 'time')
#    chan2['Rx'].dims[0].attach_scale(chan2['time'])
#    chan2['Tx'].dims[0].attach_scale(chan2['time'])

    # write geolocation metadata
    cml1.create_group('geolocation')
    lon = cml1['geolocation'].create_dataset('longitude', data=[5.66, 5.67])
    lat = cml1['geolocation'].create_dataset('latitude', data=[52.54, 52.54])
    alt = cml1['geolocation'].create_dataset('altitude', data=[30, 20])
    sit = cml1['geolocation'].create_dataset('siteID', data=['siteA', 'siteB'])
    lon.dims.create_scale(sit, 'siteID')
    lon.dims[0].attach_scale(sit)
    lat.dims[0].attach_scale(sit)
    alt.dims[0].attach_scale(sit)
    lon.attrs['units'] = 'degrees east'
    lat.attrs['units'] = 'degrees north'
    alt.attrs['units'] = 'meters above sealevel'
