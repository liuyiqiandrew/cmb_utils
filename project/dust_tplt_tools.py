import numpy as np
import healpy as hp


def beam_bl(fwhm_rad, lmax):
    return hp.gauss_beam(fwhm=fwhm_rad, lmax=lmax)


def apply_S_qu(Q, U, bl, lmax):
    # map -> (E,B) alm
    almE, almB = hp.map2alm_spin([Q, U], spin=2, lmax=lmax, iter=0)
    # multiply by beam
    hp.almxfl(almE, bl, inplace=True)
    hp.almxfl(almB, bl, inplace=True)
    # back to map
    Qs, Us = hp.alm2map_spin([almE, almB], nside=hp.get_nside(Q), spin=2, lmax=lmax, verbose=False)
    return Qs, Us


