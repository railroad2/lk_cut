from __future__ import print_function
import numpy as np
import healpy as hp
import pylab as plt
import time
from pyshtools.utils import Wigner3j

from kmlike.utils import cl2dl, print_debug, print_message
from kmlike.spectrum import get_spectrum_camb

def get_XiTT(l, lp, Wl):
    w3j, lmin, lmax = Wigner3j(l, lp, 0, 0, 0)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if lmax > lmin:
        l3 = np.arange(lmin, lmax+1)
        try:
            XiTT = sum((2*l3 + 1)/(4*np.pi) * Wl[l3] * wigwig)
        except:
            print(l3)
            XiTT = 0
    else:
        XiTT = 0

    return XiTT

def get_XiTE(l, lp, Wl):
    w3j, lmin, lmax = Wigner3j(l, lp, 0, 0, 0)
    w3j2, lmin2, lmax2 = Wigner3j(l, lp, 0, -2, 2)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if lmax > lmin:
        l3 = np.arange(lmin, lmax+1)
        zt = 1+(-1)**(l3+l+lp)
        try:    
            XiTE = sum((2*l3 + 1)/(8*np.pi) * Wl[l3] * zt * wigwig)
        except:
            print(l3)
            XiTE=0
    else:
        XiTE = 0

    return XiTE

def get_XiEE(l, lp, Wl):
    w3j, lmin, lmax = Wigner3j(l, lp, 0, -2, 2)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if lmax > lmin:
        l3 = np.arange(lmin, lmax+1)
        #l3 = np.array(list(range(lmin, lmax)))
        zt = 1+(-1)**(l3+l+lp)
        try:
            XiEE = sum((2*l3 + 1)/(16*np.pi) * Wl[l3] * zt**2 * wigwig)
        except:
            print(l3)
            XiEE= 0
    else:
        XiEE = 0

    return XiEE

def get_XiEB(l, lp, Wl):
    w3j, lmin, lmax = Wigner3j(l, lp, 0, -2, 2)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if lmax > lmin:
        l3 = np.arange(lmin, lmax+1)
        zt = 1-(-1)**(l3+l+lp)
        try:
            XiEB = sum((2*l3 + 1)/(16*np.pi) * Wl[l3] * zt**2 * wigwig)
        except:
            print(l3)
            XiEB=0
    else:
        XiEB = 0

    return XiEB

def coupling_matrix(Wl, lmin=0, lmax=100):
    WlTT = Wl
    WlPT = WlPP = WlTT

    MllTT = np.zeros((lmax, lmax))
    MllTE = np.zeros((lmax, lmax))
    MllEE = np.zeros((lmax, lmax))
    MllEB = np.zeros((lmax, lmax))

    for l in np.arange(lmax):
        for lp in np.arange(lmax):
            XiTT = get_XiTT(l, lp, WlTT) 
            XiTE = get_XiTE(l, lp, WlPT)
            XiEE = get_XiEE(l, lp, WlPP)
            XiEB = get_XiEB(l, lp, WlPP)

            MllTT[l,lp] = (2*lp + 1) * XiTT
            MllTE[l,lp] = (2*lp + 1) * XiTE
            MllEE[l,lp] = (2*lp + 1) * XiEE
            MllEB[l,lp] = (2*lp + 1) * XiEB

    MllTT = MllTT[lmin:,lmin:]
    MllTE = MllTE[lmin:,lmin:]
    MllEE = MllEE[lmin:,lmin:]
    MllEB = MllEB[lmin:,lmin:]
    M0 = np.zeros(MllTT.shape)

    Mll = [[MllTT, M0,    M0   , M0   ], 
           [M0,    MllEE, MllEB, M0   ],
           [M0,    MllEB, MllEE, M0   ],
           [M0,    M0,    M0   , MllTE]]

    Mll = np.concatenate(np.concatenate(Mll, 1), 1)

    return Mll

def EQU2GAL(m_equ):
    npix = len(m_equ)
    nside = hp.npix2nside(npix) 

    m_gal = np.full(npix, hp.UNSEEN)
    r = hp.Rotator(coord=['G', 'C'])
    n_gal = np.arange(npix)

    v_gal = hp.pix2vec(nside, n_gal)
    v_equ = r(v_gal)
    n_equ = hp.vec2pix(nside, v_equ[0], v_equ[1], v_equ[2])
    m_gal[n_gal] = m_equ[n_equ]
        
    return m_gal

def GAL2EQU(m_gal):
    npix = len(m_gal)
    nside = hp.npix2nside(npix) 

    m_equ = np.full(npix, hp.UNSEEN)
    r = hp.Rotator(coord=['C', 'G'])
    n_equ = np.arange(npix)

    v_equ = hp.pix2vec(nside, n_equ)
    v_gal = r(v_equ)
    n_gal = hp.vec2pix(nside, v_gal[0], v_gal[1], v_gal[2])
    m_equ[n_equ] = m_gal[n_gal]
        
    return m_equ

def hitmap2mask(hitmap):
    mask = np.zeros(np.shape(hitmap))

    for i, n in enumerate(hitmap):
        if n > 0:
            mask[i] = 1
        else:
            mask[i] = 0

    return mask

def cutcov_approx():
    pass     

def main():
    st = time.time()
    maskGalactic = hp.read_map('./masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field=3)
    maskPoint = hp.read_map('./masks/HFI_Mask_PointSrc_2048_R2.00.fits')
    hitGB = hp.read_map('./masks/hitMap_GB220_1024.fits')
    print ('time for reading masks', time.time() - st)

    st = time.time()
    maskGB_equ = hitmap2mask(hitGB)
    maskGB_gal = EQU2GAL(maskGB_equ)
    maskTr_equ = np.full(maskGB_equ.shape, 1)
    maskGalactic = hp.ud_grade(maskGalactic, 1024)
    print ('time for manipulating masks', time.time() - st)


    m_comb = maskGB_gal * maskGalactic
    m_comb_equ = GAL2EQU(m_comb)

    """
    hp.mollview(maskGalactic)
    hp.mollview(maskPoint)
    hp.mollview(maskGB_equ)
    hp.mollview(maskGB_gal)
    hp.mollview(m_comb)
    hp.mollview(m_comb_equ)
    """

    mask = maskGalactic
    mask = maskGB_equ

    st = time.time()
    Wl = hp.anafast(mask)
    print ('time for anafast Wl', time.time() - st)

    st = time.time()
    Wl_Tr = hp.anafast(maskTr_equ)
    print ('time for anafast Wl_Tr', time.time() - st)

    #
    Cl_camb = get_spectrum_camb(lmax=1000, isDl=False)

    np.random.seed(42)
    m = hp.synfast(Cl_camb, nside=1024, new=True)
    #
    Cl_full = hp.anafast(m)[:4]

    mm = hp.ma(m)
    mm.mask = np.logical_not(mask)

    #
    Cl_gbcut = hp.anafast(mm.filled())

    
    st = time.time()
    Mll = coupling_matrix(Wl)
    print ('time to calculate coupling matrix', time.time() - st)

    st = time.time()
    Mll_Tr = coupling_matrix(Wl_Tr)
    print ('time to calculate coupling matrix trivial', time.time() - st)

    plt.matshow(np.log10(Mll))
    plt.matshow(np.log10(Mll_Tr))
    
    st = time.time()
    Mlli = np.linalg.pinv(Mll)
    print ('time to calculate inverting coupling matrix', time.time() - st)
    plt.matshow(np.log10(Mlli))

    # 
    Cl_ana = Cl_camb[:,:100]
    Cl_ana = np.concatenate(Cl_ana, 0)
    Cl_ana_gbcut = np.dot(Mll, Cl_ana)
    Cl_ana_gbcut_inv = np.dot(Mlli, Cl_ana_gbcut)
    Cl_ana_gbcut = Cl_ana_gbcut.reshape(4, 100) #len(Cls_gbcut)/4)
    Cl_ana_gbcut_inv = Cl_ana_gbcut_inv.reshape(4, 100)
    Cl_ana_Tr = np.dot(Mll_Tr, Cl_ana)
    Cl_ana_Tr = Cl_ana_Tr.reshape(4, 100) #len(Cls_gbcut)/4)

    Cl_gbcut_tmp = Cl_gbcut[:4,:100]
    Cl_gbcut_tmp = np.concatenate(Cl_gbcut_tmp, 0)
    Cl_gbcut_inv = np.dot(Mlli, Cl_gbcut_tmp)
    Cl_gbcut_inv = Cl_gbcut_inv.reshape(4, 100)

    ell = np.arange(2, 100)

    plt.figure()
    plt.loglog(ell, cl2dl(Cl_gbcut)[:3,2:100].T, '*')
    plt.loglog(ell, cl2dl(Cl_ana_gbcut)[:3, 2:100].T)
    plt.xlabel('Multipole moment, $l$')
    plt.ylabel('$\l(l+1)/2\pi C_l (K^2)$')

    plt.figure()
    plt.loglog(ell, cl2dl(Cl_full)[:3,2:100].T, '*')
    plt.loglog(ell, cl2dl(Cl_camb)[:3,2:100].T) 
    plt.loglog(ell, cl2dl(Cl_ana_Tr)[:3,2:100].T) 
    plt.xlabel('Multipole moment, $l$')
    plt.ylabel('$\l(l+1)/2\pi C_l (K^2)$')

    plt.figure()
    plt.loglog(ell, cl2dl(Cl_gbcut_inv)[:3, 2:100].T, '*')
    plt.loglog(ell, cl2dl(Cl_ana_gbcut_inv)[:3, 2:100].T)
    plt.xlabel('Multipole moment, $l$')
    plt.ylabel('$\l(l+1)/2\pi C_l (K^2)$')

    plt.show()

if __name__=='__main__':
    main()

