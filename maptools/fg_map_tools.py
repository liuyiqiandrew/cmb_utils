import numpy as np
import pygsm

def create_mbb_scale_array(nus, beta_map, T_d=19.6, nu0=353.):
    """ create scaling array for MBB """
    scale_arr = np.ones((nus.shape[0], beta_map.shape[0]))
    scale_arr *= (nus / nu0)[:, None]
    scale_arr **= beta_map
    scale_arr *= (pygsm.planck_law(T_d, nus) / pygsm.planck_law(T_d, nu0))[:, None]
    return scale_arr