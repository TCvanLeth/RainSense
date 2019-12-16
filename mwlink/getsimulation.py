#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:48:25 2018

@author: tcvanleth
"""

from netCDF4 import Dataset

path = "/home/tcvanleth/Data/DSD_raupach/grid_2012-11-27_04_15_30.nc"
root = Dataset(path, 'r', format='NETCDF4')
