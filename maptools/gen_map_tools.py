import numpy as np
import healpy as hp
from .. import cltools


def create_map_from_power_law(nside, amp, alpha, monopole=0., lmin=2):
    """ create a spin-0 map from power law """
    ell, _, dl2cl = cltools.generate_ell_infos(nside * 4)
    bd_dl = np.zeros_like(ell, dtype=np.float64)
    bd_dl[lmin:] = amp * (ell[lmin:] / 80.)**alpha
    bd_cl = bd_dl * dl2cl
    bd_map = hp.synfast(bd_cl, nside) + monopole
    return bd_map


def create_map_from_broken_power_law(nside, amp1, alpha1, amp2, alpha2, ltrans=20, monopole=0., lmin=2):
    """ create a spin-0 map from power law """
    ell, _, dl2cl = cltools.generate_ell_infos(nside * 4)
    bd_dl = np.zeros_like(ell, dtype=np.float64)
    bd_dl[ltrans:] = amp1 * (ell[ltrans:] / 80.)**alpha1
    bd_dl[lmin:ltrans] = amp2 * (ell[lmin:ltrans] / 80.)**alpha2
    bd_cl = bd_dl * dl2cl
    bd_map = hp.synfast(bd_cl, nside) + monopole
    return bd_map