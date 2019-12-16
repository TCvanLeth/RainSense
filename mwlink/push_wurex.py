# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:02:49 2015

@author: T.C. van Leth
TO-DO: define stratiform and convective precipitation
TO-DO: catch logger
"""

import logging

from mwlink.fetch import fetch_pars
#from mwlink.fetch import fetch_radar
#from mwlink.fetch import fetch_veen
#from mwlink.fetch import fetch_pluvio
#from mwlink.fetch import fetch_wurex
from mwlink.process.atmo import proces_atmo
from mwlink.process.disdro import proces_disdro
from mwlink.process.link import proces_link


def push():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not len(logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    setID = 'WURex14'
    aux_setID = 'Veenkampen'
    radar_setID = 'KNMI'
    aux_proID = 'atmo_test1'
    DSD_proID = 'pars_test1'
    mwl_proID = 'link_test1'
    scat_ID = 'scat_test1'

    # download unprocessed data from remote sources
#    logging.info('downloading data from KNMI')
#    downloader.download_knmi(setID)
#    logging.info('downloading data from EPFL')
#    downloader.download_swiss(setID)
#    logging.info('downloading veenkampen data')
#    downloader.download_Veenkampen(aux_setID)

    # convert data to hdf5 format including metadata
#    logging.info('assimilating veenkampen data')
#    fetch_veen.fetch_veen()
    logging.info('assimilating parsivel data')
    fetch_pars.fetch_disdro()
#    logging.info('assimilating link data')
#    fetch_wurex.fetch_link()
#    logging.info('assimilating rain gauge data')
#    fetch_pluvio.fetch_pluvio()

    # fetch_radar.Radar_fetcher(radar_setID).fetch()

    # process data
#    logging.info('processing veenkampen data')
#    proces_atmo.proc_atmo_l2(aux_setID, aux_proID)

    logging.info('processing parsivel data')
    proces_disdro.proc_pars_l2(setID, DSD_proID, aux_setID, aux_proID, scat_ID)

    logging.info('aggregating parsivel data to linkpath')
    proces_disdro.proc_path_int(setID, DSD_proID)
    proces_disdro.attach_to_linkset(setID, DSD_proID, setID)

    logging.info('processing link data')
    proces_link.proc_link_l2(setID, setID, mwl_proID, DSD_proID)
    logging.info('all done!')


if __name__ == '__main__':
    push()
