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


def _ell_window(lmax: int, ell0: float, dell: float, mode: str = "C2") -> np.ndarray:
    """
    High-pass harmonic window in ell with smooth transition.

    Returns f_ell in [0,1] such that:
      f=0 for ell <= ell0-dell
      f=1 for ell >= ell0+dell
      smooth on [ell0-dell, ell0+dell]

    mode: 'C1' (raised cosine) or 'C2' (quintic smoothstep)
    """
    ell = np.arange(lmax + 1, dtype=np.float64)

    if dell <= 0:
        # hard step at ell0
        return (ell > ell0).astype(np.float64)

    x = (ell - ell0) / float(dell)           # -1..+1 in transition
    t = np.clip((x + 1.0) * 0.5, 0.0, 1.0)   # 0..1

    mode_u = mode.upper()
    if mode_u == "C1":
        s = 0.5 - 0.5 * np.cos(np.pi * t)  # C1
    elif mode_u == "C2":
        s = t**3 * (10.0 + t * (-15.0 + 6.0 * t))  # C2
    else:
        raise ValueError("mode must be 'C1' or 'C2'")
    return s
    

def highpass_ell_iqu_smooth(
    maps: np.ndarray,
    ell0: float,
    dell: float = 0.0,
    *,
    lmax: int | None = None,
    iter: int = 0,
    mode: str = "C2",
    apply_T: bool = True,
    return_alms: bool = False,
):
    """
    High-pass filter an IQU HEALPix map (or T-only) by cutting low multipoles.

    Parameters
    ----------
    maps : array
        Shape (3, npix) for (I,Q,U) or shape (npix,) for I-only.
    ell0 : float
        Cut center multipole.
    dell : float
        Half-width of smooth transition in ell. Use 0 for hard cut.
    lmax : int or None
        Max ell for harmonic transforms. Default: 3*nside-1.
    iter : int
        map2alm iterations (0 is usually fine).
    mode : {'C1','C2'}
        Smoothness of transition if dell>0.
    apply_T : bool
        If True, apply the same filter to I (T) as well. If False, leave I unchanged.
    return_alms : bool
        If True, also return filtered alms and f_ell.

    Returns
    -------
    maps_filt : array
        Filtered map, same shape as input.
    f_ell : array, optional
        The ell window used, shape (lmax+1,).
    alms_filt : tuple, optional
        Filtered (almT, almE, almB) if input is IQU, or (almT,) if I-only.
    """
    maps = np.asarray(maps)
    if maps.ndim == 1:
        # T-only
        nside = hp.get_nside(maps)
        if lmax is None:
            lmax = 3 * nside - 1
        aT = hp.map2alm(maps, lmax=lmax, iter=iter)
        f = _ell_window(lmax, ell0, dell, mode=mode)
        hp.almxfl(aT, f, inplace=True)
        mout = hp.alm2map(aT, nside=nside, lmax=lmax, verbose=False)
        if return_alms:
            return mout, f, (aT,)
        return mout

    if maps.ndim != 2 or maps.shape[0] != 3:
        raise ValueError("maps must have shape (npix,), or (3, npix) for (I,Q,U).")

    I, Q, U = maps
    nside = hp.get_nside(I)
    if lmax is None:
        lmax = 3 * nside - 1

    almT, almE, almB = hp.map2alm([I, Q, U], lmax=lmax, iter=iter, pol=True)
    f = _ell_window(lmax, ell0, dell, mode=mode)

    if apply_T:
        hp.almxfl(almT, f, inplace=True)
    hp.almxfl(almE, f, inplace=True)
    hp.almxfl(almB, f, inplace=True)

    I2, Q2, U2 = hp.alm2map([almT, almE, almB], nside=nside, lmax=lmax, pol=True, verbose=False)
    out = np.stack([I2, Q2, U2], axis=0)

    if return_alms:
        return out, f, (almT, almE, almB)
    return out


def _m_window(mmax: int, m0: float, dm: float, mode: str = "C2", kind: str = "highpass") -> np.ndarray:
    """
    Window w(m) in [0,1] over m = 0..mmax.

    highpass: suppress low m (w~0 for m<m0, w~1 for m>m0)
    lowpass : keep low m (w~1 for m<m0, w~0 for m>m0)

    If dm<=0 -> hard step.
    """
    m = np.arange(mmax + 1, dtype=np.float64)

    if dm <= 0:
        w = (m > m0).astype(np.float64)  # hard high-pass
    else:
        x = (m - m0) / float(dm)           # -1..+1 across transition
        t = np.clip((x + 1.0) * 0.5, 0.0, 1.0)

        mode_u = mode.upper()
        if mode_u == "C1":
            w = 0.5 - 0.5 * np.cos(np.pi * t)  # raised cosine (C1)
        elif mode_u == "C2":
            w = t**3 * (10.0 + t * (-15.0 + 6.0 * t))  # quintic smoothstep (C2)
        else:
            raise ValueError("mode must be 'C1' or 'C2'")

    kind_l = kind.lower()
    if kind_l == "highpass":
        return w
    if kind_l == "lowpass":
        return 1.0 - w
    raise ValueError("kind must be 'highpass' or 'lowpass'")


def hp_mcut_iqu_smooth(
    maps: np.ndarray,
    m0: float,
    dm: float = 0.0,
    *,
    lmax: int | None = None,
    iter: int = 0,
    mode: str = "C2",
    kind: str = "highpass",
    apply_T: bool = True,
    return_alms: bool = False,
):
    """
    Apply an m-cut (optionally smooth) to a HEALPix map in harmonic space.

    Input:
      - maps shape (npix,)  -> T-only
      - maps shape (3,npix) -> (I,Q,U), implemented via (T,E,B) alms

    The cut is applied as: alm_{l m} <- alm_{l m} * w(m)  (for m>=0; real-map symmetry implied)

    Parameters
    ----------
    m0 : float
        Cut center in m.
    dm : float
        Half-width of smooth transition in m. dm=0 gives a hard step.
    kind : {'highpass','lowpass'}
        highpass removes low m (keeps high m). lowpass keeps low m.
    apply_T : bool
        If True, apply the same m-window to I/T as well. If False, leave I unchanged (for IQU input).
    return_alms : bool
        If True, return filtered alms and w(m).

    Returns
    -------
    maps_filt : array
        Same shape as input.
    w_m : array, optional
        Window over m=0..lmax.
    alms : tuple, optional
        (almT,) for T-only or (almT, almE, almB) for IQU input.
    """
    maps = np.asarray(maps)

    def _apply_mwindow_inplace(alm, w_m, lmax):
        # healpy stores only m>=0 in packed format. For fixed m, l runs contiguously.
        for m in range(lmax + 1):
            w = w_m[m]
            if w == 1.0:
                continue
            i0 = hp.Alm.getidx(lmax, m, m)
            i1 = hp.Alm.getidx(lmax, lmax, m)
            alm[i0:i1 + 1] *= w

    if maps.ndim == 1:
        nside = hp.get_nside(maps)
        if lmax is None:
            lmax = 3 * nside - 1
        aT = hp.map2alm(maps, lmax=lmax, iter=iter)

        w_m = _m_window(lmax, m0, dm, mode=mode, kind=kind)
        _apply_mwindow_inplace(aT, w_m, lmax)

        mout = hp.alm2map(aT, nside=nside, lmax=lmax, verbose=False)
        if return_alms:
            return mout, w_m, (aT,)
        return mout

    if maps.ndim != 2 or maps.shape[0] != 3:
        raise ValueError("maps must have shape (npix,) or (3, npix) for (I,Q,U).")

    I, Q, U = maps
    nside = hp.get_nside(I)
    if lmax is None:
        lmax = 3 * nside - 1

    almT, almE, almB = hp.map2alm([I, Q, U], lmax=lmax, iter=iter, pol=True)

    w_m = _m_window(lmax, m0, dm, mode=mode, kind=kind)
    if apply_T:
        _apply_mwindow_inplace(almT, w_m, lmax)
    _apply_mwindow_inplace(almE, w_m, lmax)
    _apply_mwindow_inplace(almB, w_m, lmax)

    I2, Q2, U2 = hp.alm2map([almT, almE, almB], nside=nside, lmax=lmax, pol=True, verbose=False)
    out = np.stack([I2, Q2, U2], axis=0)

    if return_alms:
        return out, w_m, (almT, almE, almB)
    return out