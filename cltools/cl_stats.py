import numpy as np


def knox_covar(cl13, cl24, cl14, cl23, fsky, e_l, bin_wth):
    """
    Calculate covariance according to Knox Formula
    """
    return (cl13 * cl24 + cl14 * cl23) / (2 * e_l + 1) / fsky / bin_wth


def cl_correlation(cl11, cl22, cl12):
    """
    Calculate cl correlation between two cl
    """
    return cl12 / np.sqrt(cl11 * cl22)


