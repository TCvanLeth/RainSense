# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:42:11 2015

@author: T.C. van Leth
"""
import pyhad as ph
from mwlink import inout as io
import rainsense.disdro.disdro_level2 as dl2


def proc_pwlaw(setID, proID):
    # Parsivel DSD
    data = io.import_ds('link_aux', setID, proID)
    data = data.sel(frequency=[26e9, 38e9, 38.1e9], how='nearest')
    data = data.sel(time=slice('2015-04-01', '2015-04-02'))

    rain = data.aselect(name='h_comp').sel(htype='rain')
    print(data)
    print(rain)
    raise Exception
    data = data.mask(rain == False)

    NDpath = data.aselect(name='N(D)')
    R = data.aselect(quantity='volume_flux')

    # T-matrix scattering cross sections
    scatID = 'thurai_full_regfreq' # <-- drop shape according to Thurai 2007
    scat = io.import_ds('scatter', 'simulated', scatID, conform=True)
    scat = scat.sel(temperature=288)
    scat = scat.sel(diameter=slice(2e-4, 8e-3))

    dpc = scat.aselect(name='dpc').rechunk({'frequency': 1})
    ext = scat.aselect(name='ext').rechunk({'frequency': 1})

    # Interpolation of dsd
    NDpath, dD = dl2.resample_dsd(NDpath['parsivel', 'channel_1'], scat)

    # Attenuation and diferential phase shift forward modeling
    k = dl2.get_atten(NDpath, ext, dD)
    phi = dl2.get_diffphase(NDpath, dpc, dD)

    #k = k.mask(R <= 0.1)
    #phi = phi.mask(R <= 0.1)
    #R = R.mask(R <= 0.1)

    # Calculate power laws
    k_R = get_powerlaw(k, R, 'specific_attenuation', 'k_R')
    phi_R = get_powerlaw(phi, R, 'differential_phase_angle', 'phi_R')
    pwlaws = ph.merge([k_R, phi_R])
    print(pwlaws)
    pwlaws.attrs['level'] = 'powerlaw'
    pwlaws.name = setID
    pwlaws.attrs['pro_id'] = proID
    io.export_ds(pwlaws)


def get_powerlaw(x, y, quant, name):
    groupdims = ('frequency', 'polarization')
    x = x.groupby(*groupdims)
    pwlaws = []
    for IDs, ix in x:
        IDs = dict(IDs)

        pars = ph.analysis.powerlaw(ix, y)
        pars.setattrs(quantity=quant)
        pars.setattrs(**IDs)
        pars.setname(name, include=IDs.keys())
        pwlaws.append(pars)
    return ph.merge(pwlaws)


if __name__ == '__main__':
    setID = 'WURex14'
    proID = 'htype_algo_test2_thurai'

    proc_pwlaw(setID, proID)
