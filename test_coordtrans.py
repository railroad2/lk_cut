import numpy as np
import healpy as hp
import pylab as plt

def EQU2GAL_E(m_equ):
    npix = len(m_equ)
    nside = hp.npix2nside(npix) 
    m_gal = np.full(npix, hp.UNSEEN)
    r = hp.Rotator(coord=['C', 'G'])
    n_equ = np.arange(npix)
    v_equ = hp.pix2vec(nside, n_equ)
    v_gal = r(v_equ)
    n_gal = hp.vec2pix(nside, v_gal[0], v_gal[1], v_gal[2])
    m_gal[n_gal] = m_equ[n_equ]
    
    return m_gal

def EQU2GAL_G(m_equ):
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

def EQU2GAL_EA(m_equ):
    npix = len(m_equ)
    nside = hp.npix2nside(npix) 
    m_gal = np.full(npix, hp.UNSEEN)
    r = hp.Rotator(coord=['C', 'G'])
    n_equ = np.arange(npix)
    a_equ = hp.pix2ang(nside, n_equ)
    a_gal = r(a_equ)
    n_gal = hp.ang2pix(nside, a_gal[0], a_gal[1])
    m_gal[n_gal] = m_equ[n_equ]
    
    return m_gal

def EQU2GAL_GA(m_equ):
    npix = len(m_equ)
    nside = hp.npix2nside(npix) 
    m_gal = np.full(npix, hp.UNSEEN)
    r = hp.Rotator(coord=['G', 'C'])
    n_gal = np.arange(npix)
    a_gal = hp.pix2ang(nside, n_gal)
    a_equ = r(a_gal)
    n_equ = hp.ang2pix(nside, a_equ[0], a_equ[1])
    m_gal[n_gal] = m_equ[n_equ]

    return m_gal

if __name__=='__main__':
    nside = 2048
    npix = hp.nside2npix(nside)
    m_equ = np.arange(npix)

    m1 = EQU2GAL_E(m_equ)
    m2 = EQU2GAL_G(m_equ)
    m3 = EQU2GAL_EA(m_equ)
    m4 = EQU2GAL_GA(m_equ)

    hp.mollview(m1)
    hp.mollview(m2)
    hp.mollview(m3)
    hp.mollview(m4)

    dm = (m1==m2)*1
    dm2 = (m3==m4)*1

    hp.mollview(dm)
    hp.mollview(dm2)

    print (sum(dm), '/', npix, '=', 1.0*sum(dm)/npix)
    print (sum(dm2), '/', npix, '=', 1.0*sum(dm2)/npix)

    print (sum(m_equ))
    print (sum(m1))
    print (sum(m2))
    print (sum(m3))
    print (sum(m4))

    plt.show()

