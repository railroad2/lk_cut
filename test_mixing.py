import numpy as np
import healpy as hp
import pylab as plt

from kmlike.spectrum import get_spectrum_camb

def doit():
    nside = 512
    lmax = 3*nside-1
    
    cl = get_spectrum_camb(lmax=2000, isDl=False) 

    # maps to be tested
    m = hp.synfast(cl, nside=nside, new=True) 
    m_top = m_bot = np.zeros(m.shape)
    m_top[:,:int(len(m))] = m[:,:int(len(m))]
    m_bot[:,int(len(m)):] = m[:,int(len(m)):]

    hp.mollview(m[0])
    hp.mollview(m_top[0])
    hp.mollview(m_bot[0])

    plt.show()

    cl_full = hp.anafast(m, lmax=lmax)

if __name__ == '__main__':
    doit()

