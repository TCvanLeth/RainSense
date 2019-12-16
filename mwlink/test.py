# -*- coding: utf-8 -*-
from mwlink import inout as io

data6 = io.import_ds('pars_l2', 'WURex14', proID='pars_test6')
data5 = io.import_ds('pars_l2', 'WURex14', proID='pars_test5')
data4 = io.import_ds('pars_l2', 'WURex14', proID='pars_test4')
l1dat = io.import_ds('pars_l1', 'WURex14')

dat6 = data6['forum_2', 'channel_1']
dat5 = data5['forum_2', 'channel_1']
dat4 = data4['forum_2', 'channel_1']
datl1 = l1dat['forum_2', 'channel_1']

p6 = dat6['precip']*dat6['hmatrix'].sel(htype='rain')
p5 = dat5['precip']*dat5['hmatrix'].sel(htype='rain')
p4 = dat4['precip']*dat4['hmatrix'].sel(htype='rain')
pl1 = datl1['precip_old']*dat4['hmatrix'].sel(htype='rain')
k6 = dat6['k'].sel(frequency=38e9, polarization='H')*dat6['hmatrix'].sel(htype='rain')
k5 = dat5['k'].sel(frequency=38e9, polarization='H')*dat5['hmatrix'].sel(htype='rain')
k4 = dat4['k'].sel(frequency=38e9, polarization='H')*dat4['hmatrix'].sel(htype='rain')