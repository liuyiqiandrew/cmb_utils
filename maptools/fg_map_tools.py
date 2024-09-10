import numpy as np
import pygsm


def create_pl_scale_array(nus, beta_map, nu0=23.):
    """ create scaling array for power law """
    scale_arr = np.ones((nus.shape[0], beta_map.shape[0]))
    scale_arr *= (nus / nu0)[:, None]
    scale_arr **= beta_map
    return scale_arr


def create_mbb_scale_array(nus, beta_map, T_d=19.6, nu0=353.):
    """ create scaling array for MBB """
    scale_arr = create_pl_scale_array(nus=nus, beta_map=beta_map, nu0=nu0)
    scale_arr *= (pygsm.planck_law(T_d, nus) / pygsm.planck_law(T_d, nu0))[:, None]
    return scale_arr


def scale_maps(pivot_map, scale_arr, nus, nu0):
    """ scale the input pivoted map """
    if pivot_map.shape[0] != 1 and pivot_map.shape[0] != 2:
        raise RuntimeError(f"Pivot map have axis 0 size {pivot_map.shape[0]}, should be 1 or 2")
    pivot_map *= pygsm.tcmb2trj(nu0)
    if pivot_map.shape[0] == 2:
        scale_arr = np.repeat(scale_arr, repeats=2, axis=0).reshape(nus.shape[0], 2, pivot_map.shape[1])
    scaled_maps = pivot_map * scale_arr
    scaled_maps *= pygsm.trj2tcmb(nus)[:, None, None]
    return scaled_maps
