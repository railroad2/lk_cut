import numpy as np
import healpy as hp
import pylab as plt

from kmlike.spectrum import get_spectrum_camb
from kmlike.utils import cl2dl

def doit():
    nside = 1024
    lmax = 3*nside-1
    
    cl = get_spectrum_camb(lmax=2000, isDl=False) 
    m = hp.synfast(cl, nside=nside, new=True, verbose=False) 

    ## maps to be tested
    half = int(len(m[0])/2)
    m_t0 = np.full(m.shape, hp.UNSEEN)
    m_0b = np.full(m.shape, hp.UNSEEN)
    m_tt = np.full(m.shape, hp.UNSEEN)
    m_bb = np.full(m.shape, hp.UNSEEN)
    m_tf = np.full(m.shape, hp.UNSEEN)
    m_bp = np.full(m.shape, hp.UNSEEN)

    print (m_t0.shape)
    print (m_0b.shape)
    print (m_tt.shape)
    print (m_bb.shape)
    print (m_tf.shape)
    print (m_bp.shape)

    hm_t = m[:, :half]
    hm_b = m[:, half:]
    m_t0[:, :half] = hm_t
    m_0b[:, half:] = hm_b
    m_tt[:, :half] = hm_t
    m_tt[:, half:] = hm_t
    m_bb[:, :half] = hm_b
    m_bb[:, half:] = hm_b 
    m_tf[:, :half] = hm_t
    m_tf[:, half:] = hm_t[:,::-1]
    m_bp[:, :half] = hm_b[:,::-1]
    m_bp[:, half:] = hm_b

    ## show the maps
    hp.mollview(m[0])
    hp.mollview(m_t0[0])
    hp.mollview(m_0b[0])
    hp.mollview(m_tt[0])
    hp.mollview(m_bb[0])
    hp.mollview(m_tf[0])
    hp.mollview(m_bp[0])

    cl_full = hp.anafast(m, lmax=lmax)
    cl_t0 = hp.anafast(m_t0, lmax=lmax)
    cl_0b = hp.anafast(m_0b, lmax=lmax)
    cl_tt = hp.anafast(m_tt, lmax=lmax)
    cl_bb = hp.anafast(m_bb, lmax=lmax)
    cl_tf = hp.anafast(m_tf, lmax=lmax)
    cl_bp = hp.anafast(m_bp, lmax=lmax)

    plt.figure()
    plt.loglog(cl2dl(cl_full[:3].T))
    
    plt.figure()
    plt.loglog(cl2dl(cl_t0[:3].T))
    
    plt.figure()
    plt.loglog(cl2dl(cl_0b[:3].T))

    plt.figure()
    plt.loglog(cl2dl(cl_tt[:3].T))

    plt.figure()
    plt.loglog(cl2dl(cl_bb[:3].T))

    plt.figure()
    plt.loglog(cl2dl(cl_tf[:3].T))

    plt.figure()
    plt.loglog(cl2dl(cl_bp[:3].T))

    plt.show()

if __name__ == '__main__':
    doit()

