# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:58:43 2015

@author: T.C. van Leth
"""

import numpy as np
import xray

from . import level1c as l1
from . import inout as io
from . import filehandler as fh
from . import plotting as pt
from . import geometry as geo

datapath = 'E:/WURex14'
dirs = fh.dirstruct(datapath)

# load data
l1bdat = io.extract(dirs.l1b_path, '/Biotechnion_Forum/Forum')
dsddat = io.extract(dirs.dd_path, '/')
linkstat = io.extract(dirs.sp_path, '/Biotechnion_Forum/Forum')
linkstat2 = io.extract(dirs.sp_path, '/Biotechnion_Forum/Biotechnion')
veendat = io.extract(dirs.vp_path, 'meteostation')
raddat = io.extract(dirs.ra_path, 'linkpath')
pludat = io.extract(dirs.gp_path, '/Forum')

# get path length
path = geo.get_line(linkstat['locationID'][0], linkstat2['locationID'][0])
L = np.sqrt(np.dot(path, path))/1000

# convert raw signals to attenuation
l1cdat = l1.get_l1c(l1bdat)

# combine requested data in dataframe
att = xray.Dataset()
phi = xray.Dataset()
for name in l1cdat.data_vars.keys():
    if l1cdat[name].attrs['quantity'] == 'attenuation':
        att[name] = l1cdat[name]
    elif l1cdat[name].attrs['quantity'] == 'differential_phase':
        phi[name] = l1cdat[name]

T = xray.Dataset()
T['Temp_RAL_38_R'] = linkstat['RAL_38_temp']
T['Temp_RAL_26_R'] = linkstat['RAL_26_temp']
T['Temp_Campbell'] = linkstat['logger_temp']
T['Temp_Veenkampen'] = veendat['temp_unvent_dry']
T['Temp_RAL_38_T'] = linkstat2['RAL_38_temp'].drop('record_nr')
T['Temp_RAL_26_T'] = linkstat2['RAL_26_temp'].drop('record_nr')
T['Temp_Campbell_T'] = linkstat2['logger_temp'].drop('record_nr')

R = xray.Dataset()
for loc, ds in dsddat.groupby('locationID'):
    ds = ds.drop(('locationID', 'latitude', 'longitude',
                  'altitude', 'record_nr'))
    att['k_'+loc] = ds['k_38_H']*L
    R['Rain_'+loc] = ds['R']
R['Rain_Veenkampen'] = veendat['precipitation']*60
R['Rain_radar'] = raddat['radar_rain_unadjusted']*12
R['Rain_pluvio'] = pludat['rain_rate']

# select
h = l1bdat['htype']

h1 = h.reindex_like(linkstat2)
x = T['Temp_RAL_26_R'][h == 'dry']
y = att['RAL_26GHz_horizontal_k'][h == 'dry']

# plot
pt.timeseries({'Attenuation': att, 'Temperature': T, 'rain rate': R},
              output='test')
#pt.T_k(x, y)
#pt.k_R(dsddat)

# make kde plot
#kde = pt.kde(x, y, bandwidth=0.15)
#figpath = os.path.join(dirs.plot_dir, 'intermediate', 'test')
#io.write(kde, figpath, group='/T_k_RAL_38_H', overwrite=True)
#pt.plot_kde(kde)

#pt.cross(x, y)

pt.freqplot({'Attenuation':[att['Nokia_Flexihopper_horizontal_k'],
                            att['RAL_38GHz_horizontal_k'],
                            att['RAL_26GHz_horizontal_k']]})
pt.freqplot({'Temperature':[linkstat['logger_temp']]})

#pt.freqplot({'Attenuation (dry)':att[h == 'dry'], 'Temperature':T[h == 'dry']})

## make k_R plot
#x = dsddat['R'][dsddat['htype'] == 'liquid']
#y = dsddat['k_38_H'][dsddat['htype'] == 'liquid']
#kde = pt.kde(x, y, bandwidth=0.15, log=True)
#figpath = os.path.join(dirs.plot_dir, 'intermediate', 'test2')
#io.write(kde, figpath, group='/R_k_RAL_38_H', overwrite=True)
#pt.plot_kde(kde, log=True)
