import numpy as np
import healpy as hp

def highpass_iqu_ellcut(iqu_map, ell_cut, lmax=None, pol=True, verbose=False):
    """
    High-pass filter an IQU map by removing multipoles with ell < ell_cut.

    Parameters
    ----------
    iqu_map : array-like
        IQU map as shape (3, npix) or [I, Q, U], in RING ordering.
    ell_cut : int
        Cutoff multipole. All multipoles with ell < ell_cut are set to 0.
        (ell_cut=0 => no filtering; ell_cut=2 removes monopole+dipole, etc.)
    lmax : int or None
        Maximum multipole for the harmonic transform. If None, uses 3*nside-1.
    pol : bool
        Passed to healpy (True for IQU).
    verbose : bool
        Verbosity flag passed to healpy.

    Returns
    -------
    iqu_hp : np.ndarray
        Filtered IQU map with same shape as input (3, npix).
    """
    m = np.asarray(iqu_map)
    if m.ndim != 2 or m.shape[0] != 3:
        raise ValueError("iqu_map must have shape (3, npix) (I, Q, U).")

    npix = m.shape[1]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1
    if ell_cut < 0 or ell_cut > lmax + 1:
        raise ValueError(f"ell_cut must be in [0, {lmax+1}] for this lmax.")

    # Map -> alm (T, E, B)
    almT, almE, almB = hp.map2alm(m, lmax=lmax, pol=pol, verbose=verbose)

    # Zero low-ell modes
    if ell_cut > 0:
        for alm in (almT, almE, almB):
            hp.almxfl(alm, np.array([0.0] * ell_cut + [1.0] * (lmax - ell_cut + 1)), inplace=True)

    # alm -> map
    iqu_hp = hp.alm2map([almT, almE, almB], nside=nside, lmax=lmax, pol=pol, verbose=verbose)

    # Ensure (3, npix)
    return np.asarray(iqu_hp)
