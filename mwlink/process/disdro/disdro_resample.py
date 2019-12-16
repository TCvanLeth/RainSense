# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 13:38:00 2015

@author: T.C. van Leth
"""

import logging

import numpy as np
import phad as ph
from phad import ufuncs as uf

from mwlink import inout as io


def proc_path_int(setID, proID, **kwargs):
    data = io.import_ds('pars_l2', setID, proID, **kwargs)
    data = data.sel(time=slice("2015-04-01", "2016-01-01"))
#    data2 = io.import_ds('pars_l1', setID, **kwargs)
#    data2 = data2[['forum_1', 'forum_2', 'nvwa', 'biotechnion', 'bongerd'], ['channel_1']]
    data = data[['forum_1', 'forum_2', 'nvwa', 'biotechnion', 'bongerd'], ['channel_1'], ['N(D)', 'precip', 'lwc']]
    odat = set_to_path(data)
    #odat2 = set_to_path(data2[:, :, 'visibility'])
    #tdat = odat.apply(get_htypes)
    #edat = get_event(tdat)
    #mdat = ha.merge([tdat, edat])
    print("saving")
    io.export_ds(odat, level='link_aux')


def attach_to_linkset(DSD_setID, DSD_proID, link_setID, **kwargs):
    inset = io.import_ds('link_aux', DSD_setID, DSD_proID, **kwargs)
    chan = inset['parsivel', 'channel_1', ['htypes', 'h_comp', 'event_nr']]

    outset = type(inset)()
    outset.aux.aux = chan
    io.export_ds(outset, level='link_l1', set_id=link_setID, daily=True)


###############################################################################
def set_to_path(inset):
    """
    """
    logging.info('calculating path integrated bulk quantities')
    attrs = ph.Attributes(site_a_id='biotechnion',
                          site_b_id='forum_2',
                          site_a_latitude=51.968657,
                          site_b_latitude=51.985288,
                          site_a_longitude=5.68273,
                          site_b_longitude=5.664268,
                          site_a_altitude=np.nan,
                          site_b_altitude=np.nan)
    sites = ph.Station(attrs=attrs)
    wgt = get_weight(inset, sites)
    outstat = get_int_bulk(inset, wgt)

    outstat.attrs.update(attrs)
    outstat.attrs['station_id'] = 'parsivel'
    outset = type(inset)(children={outstat.attrs['station_id']: outstat},
                         attrs=inset.attrs, name=inset.name)
    return outset


def get_weight(inset, sites):
    """
    get the weights of the station values based on the positions of
    the stations
    """
    IDs = list(inset.children.keys())
    if len(IDs) == 1:
        return ph.Array([1.], [ph.index(IDs, 'site_id')])

    # gps coordinates to distances
    l = sites.path.compute()
    lines = get_path(inset, sites).compute()
    L = np.sqrt(l @ l)
    s = (lines @ l) / L
    index = np.argsort(s)
    s = s[index]

    # weighted by relative distance
    wgt = np.zeros_like(s)
    wgt[1:-1] = 0.5 * (s[2:] - s[:-2]) / L
    wgt[0] = 0.5 * (s[1] - s[0]) / L
    wgt[-1] = 0.5 * (s[-1] - s[-2]) / L

    rindex = np.empty(len(index), dtype=np.int)
    rindex[index] = np.arange(len(index))
    wgt = wgt[rindex]
    return ph.Array(wgt, [ph.Index(IDs, 'site_id')])


def get_path(inset, sites):
    x1 = sites.attrs['site_a_longitude']
    y1 = sites.attrs['site_a_latitude']
    z1 = sites.attrs['site_a_altitude']

    x2 = np.zeros(len(inset.children))
    y2 = np.zeros(len(inset.children))
    z2 = np.zeros(len(inset.children))
    for i, istat in enumerate(inset.children.values()):
        x2[i] = istat.attrs['site_longitude']
        y2[i] = istat.attrs['site_latitude']
        z2[i] = istat.attrs['site_altitude']

    return ph.geometry.haversines(x1, y1, z1, x2, y2, z2)


def get_int_bulk(inset, wgt):
    """
    weighted average over bulk variables.
    """
    instats = list(inset.children.values())
    cstat = ph.merge(instats, xdim='site_id')
    wstat1 = uf.average(cstat.select(dtype=float), wgt, dim='site_id')
    wstd = uf.sqrt(uf.average((cstat.select(dtype=float) - wstat1)**2, wgt, dim='site_id')/(1-uf.sum(wgt**2)))
    wstd.setname('_std', append=True)
    wstat2 = uf.maximum(cstat.select(dtype=int), 0).any(dim='site_id')
    wstat2 = wstat2.astype('int')
    return ph.merge([wstat1, wstat2, wstd])


###############################################################################
def get_htypes(indat, mattrs):
    logging.info('determining precipitation type')
    hmat = indat.aselect(quantity='hydrometeor_composition')

    htypes = ph.full([hmat['time']], 'mixed', dtype='S7', name='htypes')
    for tID in hmat['htype'].values:
        itype = hmat.sel(htype=tID)
        other = hmat.drop(htype=tID)
        cond = (itype & ~other.any(dim='htype')).astype(bool)
        htypes[cond] = tID
    cond = ~hmat.any(dim='htype').astype(bool)
    htypes[cond] = 'dry'
    htypes.attrs = hmat.attrs
    htypes.setattrs(quantity='hydrometeor_type', unit='none')
    return indat.update(htypes)


def get_event(indat, delta=30):
    logging.info('determining path events')
    htypes = indat.aselect(quantity='hydrometeor_type')

    begin = uf.rollall(htypes != 'dry', delta, dim='time')
    end = uf.rollall(htypes == 'dry', delta, dim='time')
    event = uf.count(uf.hystgate(begin, end, dim='time'))

    event.setattrs(quantity='event_number', unit='none')
    event.setname('event_nr')
    return event


###############################################################################
# deprecation candidates
def precip_conv(data, mattrs):
    invars = data.select(quantity='hydrometeor_composition')

    unit = invars.getattrs['unit']
    if unit == 'SYNOP_WW_code':
        outvars = isprecip(cons_htypes(invars))
    elif unit == 'particle_class':
        outvars = isprecip(invars)
    elif unit == 'particle_class_presence':
        outvars = invars
    else:
        raise ph.units.UnitError('hydrometeor types in unknown format! "%s"' % unit)

    outvars.setattrs(unit='particle_class_presence')
    outvars.setattrs(sampling='mean')
    outvars.setname('hmatrix')
    return outvars

def isprecip(htype):
    htypes = ph.index(['rain', 'snow', 'hail', 'mix', 'graupel'], 'htype')
    return (htype == htypes).astype(int)


###############################################################################
if __name__ == '__main__':
    ph.common.standardlogger()
    setID = 'WURex14'
    DSD_proID = 'htype_algo_test2_thurai'

    proc_path_int(setID, DSD_proID)
