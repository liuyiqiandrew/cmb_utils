import healpy as hp
import numpy as np


def lat2mask(nside, lat):
    """
    Create simple mask based on latitude cut
    """
    lat_rad = lat / 180 * np.pi
    pix_ind = np.arange(hp.nside2npix(nside))
    pix_ang = np.array(hp.pix2ang(nside, pix_ind, lonlat=True)) / 180 * np.pi
    return (pix_ang[1] < lat_rad) * (pix_ang[1] > -lat_rad)


def carrbox2hpmask(nside, box):
    """
    Convert a carr box to HEALPix mask
    """
    pix_ind = np.arange(hp.nside2npix(nside))
    pix_ang = np.array(hp.pix2ang(nside, pix_ind, lonlat=True)) / 180 * np.pi
    if box[1,1] < 0:
        ra_cut = (pix_ang[0] > (2 * np.pi + box[0, 1])) * (pix_ang[0] < (2 * np.pi + box[1, 1]))
    else:
        ra_cut = (pix_ang[0] > box[0, 1]) * (pix_ang[0] < box[1, 1])
    dec_cut = (pix_ang[1] > box[0, 0]) * (pix_ang[1] < box[1, 0])
    return ra_cut * dec_cut