from __future__ import print_function
import numpy  as np
import healpy as hp
import pylab  as plt
import time

from cutsky import hitmap2mask, coupling_matrix
from cutsky_fast import coupling_matrix as coupling_matrix_fast
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
    mm = hp.ma(m)
    mm.mask = np.logical_not(mask)
    hp.mollview(mm.filled()[0])
    print_debug('Time for synfast:', time.time()-st)
    
    st = time.time()
    cl_ana = hp.anafast(m)
    print_debug('Time for get anafast:', time.time()-st)

    st = time.time()
    Wl = hp.anafast(mask)
    Mll = coupling_matrix(Wl, lmin=lmin, lmax=lmax)
    Mlli = np.linalg.pinv(Mll)
    print_debug('Time for getting Mll:', time.time()-st)

    st = time.time()
    biased = hp.anafast(mm.filled())
    print_debug('Time for getting biased spectrum:', time.time()-st)
    st = time.time()
    Cl_tmp = np.concatenate(biased[:4, lmin:lmax], 0)
    unbiased = np.dot(Mlli, Cl_tmp).reshape(4, lmax-lmin)
    print_debug('Time for get unbiased spectra:', time.time()-st)

    st = time.time()
    cl_ana = hp.anafast(m)
    print_debug('Time for get anafast:', time.time()-st)

    plt.figure()
    plt.loglog(np.abs(cl2dl(cl[:3].T)))
    plt.loglog(np.abs(cl2dl(cl_ana[:3].T)), '*-', lw=0.5)
    plt.loglog(np.abs(cl2dl(biased[:3].T)), 'x-', lw=0.5)
    plt.loglog(np.abs(cl2dl(unbiased[:3])).T, '+-', lw=0.5)
    plt.ylim(1e-6, 1e4)
    plt.xlabel = 'multipole moment ($l$)'
    plt.ylabel = '$D_l (K^2)'
    plt.title("Coupling matrix")

    return biased, unbiased

def test_ensemble(cl, mask, rseed, lmin=0, lmax=100, nside=256, nsample=100):
    print_message('Ensemble test with my code')
    print_message('Mask coverage = %f' % (np.average(mask)))
    #mask = np.logical_not(hp.ud_grade(mask, nside_out = nside))
    mask = hp.ud_grade(mask, nside_out = nside)
    print_message('Mask coverage for nside %d = %f' % (nside, np.average(mask)))

    np.random.seed(rseed)
    seeds = np.random.randint(low=0, high=2**32-1, size=nsample)

    Wl = hp.anafast(mask, lmax=3*nside)

    st = time.time()
    Mll = coupling_matrix(Wl, lmin=lmin, lmax=lmax)
    print_debug('Time for getting Mll:', time.time()-st)

    st = time.time()
    Mllp = coupling_matrix_fast(Wl, lmin=lmin, lmax=lmax)
    print_debug('Time for getting Mll_fast:', time.time()-st)

    plt.matshow(Mll)
    plt.matshow(Mllp)
    plt.matshow(Mllp - Mll)
    plt.show()

    st = time.time()
    Mlli = np.linalg.pinv(Mll)
    print_debug('Time for getting Mlli:', time.time()-st)

    biased_arr = []
    unbiased_arr = []

    for i, s in enumerate(seeds):
        print_message('sample #%d' % i)

        np.random.seed(s)
        st = time.time()
        m = hp.synfast(cl, nside=nside, new=True)
        mm = hp.ma(m)
        mm.mask = np.logical_not(mask)
        print_debug('Time for synfast:', time.time()-st)
    
        st = time.time()
        biased = hp.anafast(mm.filled(), lmax=lmax)
        print_debug('Time for getting biased spectrum:', time.time()-st)

        st = time.time()
        Cl_tmp = np.concatenate(biased[:4, lmin:lmax], 0)
        unbiased = np.dot(Mlli, Cl_tmp).reshape(4, lmax-lmin)
        print_debug('Time for get unbiased spectra:', time.time()-st)

        biased_arr.append(biased)
        unbiased_arr.append(unbiased)

        #st = time.time()
        #cl_ana = hp.anafast(m)
        #print_debug('Time for get anafast:', time.time()-st)

    biased = np.average(biased_arr, axis=0)
    unbiased_tmp = np.average(unbiased_arr, axis=0)
    unbiased = np.zeros(biased.shape)
    unbiased[:4,lmin:lmax]=unbiased_tmp

    print (biased.shape)
    print (unbiased.shape)

    plt.figure()
    plt.loglog(np.abs(cl2dl(cl[:3].T)))
    #plt.loglog(np.abs(cl2dl(cl_ana[:3].T)), '*-', lw=0.5)
    plt.loglog(np.abs(cl2dl(biased[:3].T)), 'x-', lw=0.5)
    plt.loglog(np.abs(cl2dl(unbiased[:3])).T, '+-', lw=0.5)
    plt.ylim(1e-6, 1e4)
    plt.xlabel = 'multipole moment ($l$)'
    plt.ylabel = '$D_l (K^2)'
    plt.title("Coupling matrix Ensemble")

    return biased, unbiased

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
    mask_GB = hitmap2mask(hp.read_map(fnmask[-1]))
    #hp.mollview(mask)

    rseed = 42
    test_single(cl, mask, rseed, lmin=0, lmax=200, nside=256)
    #test_ensemble(cl, mask, rseed, lmin=0, lmax=100, nside=256, nsample=10)

    plt.show()

if __name__=="__main__":
    doit()
