# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:35:34 2015

@author: T.C. van Leth
"""
import os
import yaml

from .core import filehandler as fh
from .fetch import fetch_tmobile as tf


def Tmobile():
    datapath = 'D:/TmobileNL'
    dirs = fh.dirstruct(datapath)
    path = os.path.join(os.path.dirname(__file__),
                        'Parameters', 'Tmobile.yaml')
    with open(path, mode='r') as inh:
        for meta in yaml.safe_load_all(inh):
            tf.fetch_Tlink(meta, dirs)

if __name__ == '__main__':
    Tmobile()
