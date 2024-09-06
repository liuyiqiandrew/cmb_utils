import numpy as np

def generate_ell_infos(lmax):
    """ create commonly used ell info (ell, cl2dl, dl2cl) """
    ell = np.arange(lmax)
    cl2dl = ell * (ell + 1) / 2 / np.pi
    cl2dl[:2] = 0
    dl2cl = np.zeros_like(cl2dl)
    dl2cl[2:] = 1 / cl2dl[2:]
    return ell, cl2dl, dl2cl