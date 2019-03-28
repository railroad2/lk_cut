from __future__ import print_function, absolute_import
import numpy as np
import healpy as hp
import pylab as plt
import time

import xpol
import cutsky as cs
from kmlike.utils import cl2dl, print_debug, print_message
from kmlike.spectrum import get_spectrum_camb

def test_single(cl, mask, rseed, lmin=0, lmax=100, nside=256):
    print_message('Single test with my code')
    print_message('Mask coverage = %f' % (np.average(mask)))
    #mask = np.logical_not(hp.ud_grade(mask, nside_out = nside))
    mask = hp.ud_grade(mask, nside_out = nside)
    print_message('Mask coverage for nside %d = %f' % (nside, np.average(mask)))

    np.random.seed(rseed)
    st = time.time()
    m = hp.synfast(cl, nside=nside, new=True)
    print_debug('Time for synfast:', time.time()-st)

    st = time.time()
    cl_ana = hp.anafast(m)
    print_debug('Time for get anafast:', time.time()-st)

    st = time.time()
    lmins = np.arange(lmin, lmax)
    lmaxs = lmins + 1 
    bins = xpol.Bins(lmins, lmaxs)
    xp = xpol.Xpol(mask, bins)
    print_debug('Time for initalizing x-pol:', time.time()-st)

    st = time.time()
    biased, unbiased_tmp = xp.get_spectra(m)
    print_debug('Time for get spectra from x-pol:', time.time()-st)
    
    unbiased = np.zeros(biased.shape)
    unbiased[:,2:] = unbiased_tmp

    plt.figure()
    plt.loglog(np.abs(cl2dl(cl[:3].T)))
    plt.loglog(np.abs(cl2dl(cl_ana[:3].T)), '*-', lw=0.5)
    plt.loglog(np.abs(cl2dl(biased[:3].T)), 'x-', lw=0.5)
    plt.loglog(np.abs(cl2dl(unbiased[:3])).T, '+-', lw=0.5)
    plt.ylim(1e-6, 1e4)
    plt.xlabel = 'multipole moment ($l$)'
    plt.ylabel = '$D_l (K^2)'
    plt.title("x-pol")

    return biased, unbiased

def test_ensemble(cl, mask, rseed, lmin=0, lmax=100, nside=256, nsample=100):
    print_message('Ensemble test with my code')
    print_message('Mask coverage = %f' % (np.average(mask)))
    #mask = np.logical_not(hp.ud_grade(mask, nside_out = nside))
    mask = hp.ud_grade(mask, nside_out = nside)
    print_message('Mask coverage for nside %d = %f' % (nside, np.average(mask)))

    np.random.seed(rseed)
    seeds = np.random.randint(low=0, high=2**32-1, size=nsample)

    st = time.time()
    lmins = np.arange(lmin, lmax)
    lmaxs = lmins + 1 
    bins = xpol.Bins(lmins, lmaxs)
    xp = xpol.Xpol(mask, bins)
    print_debug('Time for initalizing x-pol:', time.time()-st)


    biased_arr = []
    unbiased_arr = []

    for i, s in enumerate(seeds):
        print_message('sample #%d' % i)

        np.random.seed(s)
        st = time.time()
        m = hp.synfast(cl, nside=nside, new=True)
        print_debug('Time for synfast:', time.time()-st)
 
        st = time.time()
        biased, unbiased = xp.get_spectra(m)
        print_debug('Time for get spectra from x-pol:', time.time()-st)

        biased_arr.append(biased)
        unbiased_arr.append(unbiased)

        #st = time.time()
        #cl_ana = hp.anafast(m)
        #print_debug('Time for get anafast:', time.time()-st)

    biased = np.average(biased_arr, axis=0)
    unbiased_tmp = np.average(unbiased_arr, axis=0)
    print (biased.shape)
    print (unbiased_tmp.shape)
    unbiased = np.zeros(biased.shape)
    unbiased[:,lmin:]=unbiased_tmp

    plt.figure()
    plt.loglog(np.abs(cl2dl(cl[:3].T)))
    #plt.loglog(np.abs(cl2dl(cl_ana[:3].T)))
    plt.loglog(np.abs(cl2dl(biased[:3].T)), 'x-', lw=0.5)
    plt.loglog(np.abs(cl2dl(unbiased[:3])).T, '+-', lw=0.5)
    plt.ylim(1e-6, 1e4)
    plt.xlabel = 'multipole moment ($l$)'
    plt.ylabel = '$D_l (K^2)'
    plt.title('X-pol Ensemble')

def doit():
    fnmask=["./masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
            "./masks/COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits",
            "./masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",
            "./masks/HFI_Mask_GalPlane-apo2_2048_R2.00.fits",
            "./masks/HFI_Mask_GalPlane-apo5_2048_R2.00.fits",
            "./masks/HFI_Mask_PointSrc_2048_R2.00.fits",
            "./masks/hitMap_GB145_1024.fits",
            "./masks/hitMap_GB220_1024.fits",]

    cl = get_spectrum_camb(lmax=2000, r=0.01, isDl=False, CMB_unit='muK')

    mask = hp.read_map(fnmask[1])
    mask_GB = cs.hitmap2mask(hp.read_map(fnmask[-1]))
    hp.mollview(mask)

    rseed = 42
    test_single(cl, mask, rseed, lmin=0, lmax=100, nside=256)
    #test_ensemble(cl, mask, rseed, lmin=30, lmax=100, nside=128, nsample=100)

    plt.show()

if __name__=='__main__':
    doit()

