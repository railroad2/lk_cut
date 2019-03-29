import numpy as np
import healpy as hp
import pylab as plt

from kmlike.spectrum import get_spectrum_camb
from kmlike.utils import cl2dl

def single(seed):
    nside = 512
    lmax = 3*nside - 1
    cl = get_spectrum_camb(lmax=lmax, isDl=False)
    np.random.seed(seed)
    m = hp.synfast(cl, nside, new=True, verbose=False)
    half = int(len(m[0])/2)
    hm_top = np.zeros(m.shape)
    hm_fob = np.zeros(m.shape)
    hm_top[:,:half] = m[:,:half]
    hm_fob[:,half:] = m[:,half:]
    
    cl_top = hp.anafast(hm_top, lmax=lmax)
    cl_fob = hp.anafast(hm_fob, lmax=lmax)

    return cl_top, cl_fob
    
def ensemble(seed, nsamples=10):
    np.random.seed(seed)
    seeds = np.random.randint(0, 2**32-1, nsamples)

    cl_t = []
    cl_b = []
    for i, s in enumerate(seeds):
        print ('sample #', i)
        cl_top, cl_fob = single(s)
        cl_t.append(cl_top)
        cl_b.append(cl_fob)

    cl_t = np.average(cl_t, axis=0)
    cl_b = np.average(cl_b, axis=0)

    hp.write_cl('cl_top.fits', cl_t)
    hp.write_cl('cl_bot.fits', cl_b)

    
if __name__=="__main__":
    ensemble(42, nsamples=1000)


