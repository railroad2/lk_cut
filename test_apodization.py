from __future__ import print_function
import numpy as np
import pylab as plt
import healpy as hp

import xpol

from kmlike.spectrum import get_spectrum_camb
from kmlike.utils import cl2dl

def cl_mask_apod(cmb, mask, fwhm, nbins=100):
    coverage = np.average(mask)
    if (fwhm != 0):
        mask_apo = hp.smoothing(mask, fwhm=fwhm, verbose=False)
    else:
        mask_apo = mask
    hp.mollview(mask_apo)

    lmins = np.arange(nbins)
    lmaxs = lmins + 1
    bins = xpol.Bins(lmins, lmaxs)
    xp = xpol.Xpol(mask_apo, bins)

    bi, unbi_t = xp.get_spectra(cmb)
    bi = bi/coverage
    unbi = np.zeros(bi.shape)
    unbi[:,2:] = unbi_t

    return bi, unbi

def test_halfsky():
    nside = 512
    npix = hp.nside2npix(nside)
    lmax = nside * 3 - 1
    
    mask = np.zeros(npix)
    mask[:int(len(mask)/2.)] = 1
    cl = get_spectrum_camb(lmax=lmax, isDl=False)
    cmb = hp.synfast(cl, nside=nside, new=True, verbose=False)
    biapo, unbiapo = cl_mask_apod(cmb, mask, np.radians(10), nbins=100)
    bi, unbi = cl_mask_apod(cmb, mask, np.radians(0), nbins=100)

    plt.figure()
    plt.loglog(cl2dl(abs(cl[:3])).T)
    plt.loglog(cl2dl(abs(biapo[:3])).T, 'x-', lw=0.5)
    plt.loglog(cl2dl(abs(unbiapo[:3])).T, '+-', lw=0.5)
    plt.xlabel('$l$')
    plt.ylabel('$D_l (K^2)$')
    plt.title('Apodized mask')

    plt.figure()
    plt.loglog(cl2dl(abs(cl[:3])).T)
    plt.loglog(cl2dl(abs(bi[:3])).T, 'x-', lw=0.5)
    plt.loglog(cl2dl(abs(unbi[:3])).T, '+-', lw=0.5)
    plt.xlabel('$l$')
    plt.ylabel('$D_l (K^2)$')
    plt.title('normal mask')

    plt.show()

def test_planckmask():
    nside = 512
    npix = hp.nside2npix(nside)
    lmax = nside * 3 - 1
    mask = hp.read_map("./masks/COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits")
    mask = hp.ud_grade(mask, nside_out=nside)


    cl = get_spectrum_camb(lmax=lmax, isDl=False)
    cmb = hp.synfast(cl, nside=nside, new=True, verbose=False)
    biapo, unbiapo = cl_mask_apod(cmb, mask, np.radians(5), nbins=100)
    bi, unbi = cl_mask_apod(cmb, mask, np.radians(0), nbins=100)

    plt.figure()
    plt.loglog(cl2dl(abs(cl[:3])).T)
    plt.loglog(cl2dl(abs(biapo[:3])).T, 'x-', lw=0.5)
    plt.loglog(cl2dl(abs(unbiapo[:3])).T, '+-', lw=0.5)
    plt.xlabel('$l$')
    plt.ylabel('$D_l (K^2)$')
    plt.title('Apodized mask')

    plt.figure()
    plt.loglog(cl2dl(abs(cl[:3])).T)
    plt.loglog(cl2dl(abs(bi[:3])).T, 'x-', lw=0.5)
    plt.loglog(cl2dl(abs(unbi[:3])).T, '+-', lw=0.5)
    plt.xlabel('$l$')
    plt.ylabel('$D_l (K^2)$')
    plt.title('normal mask')

    plt.show()

if __name__=='__main__':
    #test_halfsky()
    test_planckmask()
    
