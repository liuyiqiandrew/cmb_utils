import numpy as np
import healpy as hp
from .. import cltools


def create_map_from_power_law(nside, amp, alpha, monopole=0.):
    """ create a spin-0 map from power law """
    ell, _, dl2cl = cltools.generate_ell_infos(nside * 4)
    bd_dl = np.zeros_like(ell)
    bd_dl[2:] = amp * (ell[2:] / 80.)**alpha
    bd_cl = bd_dl * dl2cl
    bd_map = hp.synfast(bd_cl, nside) + monopole
    return bd_map