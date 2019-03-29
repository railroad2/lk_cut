from __future__ import print_function, absolute_import
import numpy as np
import healpy as hp
import pylab as plt
import time
from pyshtools.utils import Wigner3j
from cutsky import get_XiTT, get_XiTE, get_XiEE, get_XiEB

def get_XiTT_f(int l, int lp, double[:] Wl):
    cdef double[:] w3j
    cdef double[:] wigwig
    cdef int lmin, lmax
    cdef long[:] l3
    cdef long ll, i
    cdef double XiTT

    w3j, lmin, lmax = Wigner3j(l, lp, 0, 0, 0)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if lmax > lmin:
        l3 = np.arange(lmin, lmax)

        XiTT = 0
        ll = len(l3)
        for i in range(ll):
            XiTT += (2*l3[i] + 1)/(4*np.pi) * Wl[i] * wigwig[i]
    else:
        XiTT = 0

    return XiTT

def get_XiTE_f(int l, int lp, double[:] Wl):
    cdef double[:] w3j, w3j2
    cdef double[:] wigwig
    cdef int lmin, lmax, lmin2, lmax2
    cdef long[:] l3
    cdef long ll, i
    cdef double XiTE, zt 


    w3j, lmin, lmax = Wigner3j(l, lp, 0, 0, 0)
    w3j2, lmin2, lmax2 = Wigner3j(l, lp, 0, -2, 2)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if (lmax > lmin):
        l3 = np.arange(lmin, lmax)

        XiTE = 0
        ll = len(l3)
        for i in range(ll):
            zt = 1+(-1)**(l3[i]+l+lp)
            XiTE += (2*l3[i] + 1)/(8*np.pi) * Wl[i] * zt * wigwig[i]
    else:
        XiTE = 0

    return XiTE

def get_XiEE_f(int l, int lp, double[:] Wl):
    cdef double[:] w3j
    cdef double[:] wigwig
    cdef int lmin, lmax
    cdef long[:] l3
    cdef long ll, i
    cdef double XiEE, zt 

    w3j, lmin, lmax = Wigner3j(l, lp, 0, -2, 2)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if (lmax > lmin):
        l3 = np.arange(lmin, lmax)

        XiEE = 0
        ll = len(l3)
        for i in range(ll):
            zt = 1+(-1)**(l3[i]+l+lp)
            XiEE += (2*l3[i] + 1)/(16*np.pi) * Wl[i] * zt**2 * wigwig[i]
    else:
        XiEE = 0

    return XiEE

def get_XiEB_f(int l, int lp, double[:] Wl):
    cdef double[:] w3j
    cdef double[:] wigwig
    cdef int lmin, lmax
    cdef long[:] l3
    cdef long ll, i
    cdef double XiEB, zt 

    w3j, lmin, lmax = Wigner3j(l, lp, 0, -2, 2)
    wigwig = np.power(w3j, 2)[:lmax-lmin+1]

    if (lmax > lmin):
        l3 = np.arange(lmin, lmax)

        XiEB = 0
        ll = len(l3)
        for i in range(ll):
            zt = 1-(-1)**(l3[i]+l+lp)
            XiEB += (2*l3[i] + 1)/(16*np.pi) * Wl[i] * zt**2 * wigwig[i]
    else:
        XiEB = 0

    return XiEB

def coupling_matrix(double[:] Wl, int lmin=0, int lmax=100):
    #cdef double[:] WlTT, WlPT, WlPP
    #cdef double[:,:] MllTT, MllTE, MllEE, MllEB
    cdef int l, lp
    #cdef double[:,:] Mll
    cdef int dim

    WlTT = WlPT = WlPP = Wl
    MllTT = MllTE = MllEE = MllEB = np.zeros((lmax, lmax))

    for l in range(lmax):
        for lp in range(lmax):
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
    #M0 = np.zeros(MllTT.shape)
    dim = MllTT.shape[0]
    Mll = np.zeros((4*dim, 4*dim))

    #Mll = [[MllTT, M0,    M0   , M0   ], 
    #       [M0,    MllEE, MllEB, M0   ],
    #       [M0,    MllEB, MllEE, M0   ],
    #       [M0,    M0,    M0   , MllTE]]
    #Mll = np.concatenate(np.concatenate(Mll, 1), 1)

    Mll[0*dim:1*dim,0*dim:1*dim] = MllTT
    Mll[3*dim:4*dim,3*dim:4*dim] = MllTE

    Mll[1*dim:2*dim,1*dim:2*dim] = MllEE
    Mll[2*dim:3*dim,2*dim:3*dim] = MllEE

    Mll[1*dim:2*dim,2*dim:3*dim] = MllEB
    Mll[2*dim:3*dim,1*dim:2*dim] = MllEB


    print(Mll.shape)


    return Mll

