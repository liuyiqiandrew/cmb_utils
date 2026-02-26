import numpy as np
import healpy as hp
from .. import cltools


def create_map_from_power_law(
    nside: int, 
    amp: float, 
    alpha: float, 
    monopole: float=0., 
    lmin: int=2, 
    flat_out: int=15
):
    """ create a spin-0 map from power law """
    ell, _, dl2cl = cltools.generate_ell_infos(nside * 4)
    bd_dl = np.zeros_like(ell, dtype=np.float64)
    bd_dl[lmin:] = amp * (ell[lmin:] / 80.)**alpha
    bd_dl[lmin:][ell[lmin:] < flat_out] = bd_dl[flat_out]
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


def generate_qu_noise(sig2_QQ, sig2_UU, sig2_QU, rng=None, eps=0.0):
    """
    Generate correlated Gaussian Q/U noise per pixel given covariance components.

    Parameters
    ----------
    sig2_QQ, sig2_UU, sig2_QU : array-like
        Arrays (same shape) containing covariance elements:
        Cov(Q,Q)=sig2_QQ, Cov(U,U)=sig2_UU, Cov(Q,U)=sig2_QU.
        (Despite the names, these are covariance elements, not necessarily squares of something.)
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().
    eps : float, optional
        Optional diagonal regularization added to QQ and UU (useful if some pixels are near-singular).

    Returns
    -------
    nQ, nU : np.ndarray
        Correlated noise realizations with the same shape as inputs.
    """
    sig2_QQ = np.asarray(sig2_QQ, dtype=float)
    sig2_UU = np.asarray(sig2_UU, dtype=float)
    sig2_QU = np.asarray(sig2_QU, dtype=float)
    if sig2_QQ.shape != sig2_UU.shape or sig2_QQ.shape != sig2_QU.shape:
        raise ValueError("sig2_QQ, sig2_UU, sig2_QU must have identical shapes")

    if rng is None:
        rng = np.random.default_rng()

    # Flatten for vectorized math; reshape back at end
    shape = sig2_QQ.shape
    qq = sig2_QQ.reshape(-1) + eps
    uu = sig2_UU.reshape(-1) + eps
    qu = sig2_QU.reshape(-1)

    # Cholesky for 2x2:
    # L = [[a, 0],
    #      [b, c]] with a=sqrt(qq), b=qu/a, c=sqrt(uu - b^2)
    # Need qq>0 and uu - (qu^2/qq) >= 0 for PSD.
    if np.any(qq <= 0):
        bad = np.where(qq <= 0)[0][:10]
        raise ValueError(f"Non-positive QQ variance in {bad.size} pixels (showing up to 10): {bad}")

    a = np.sqrt(qq)
    b = qu / a
    rad = uu - b*b
    if np.any(rad < 0):
        # allow tiny negative due to rounding; clamp those
        tol = 1e-12 * np.maximum(uu, 1.0)
        if np.any(rad < -tol):
            bad = np.where(rad < -tol)[0][:10]
            raise ValueError(
                f"Covariance not PSD in {bad.size} pixels (showing up to 10): {bad}. "
                "Try eps>0 or inspect QQ/UU/QU values."
            )
        rad = np.maximum(rad, 0.0)
    c = np.sqrt(rad)

    z0 = rng.standard_normal(qq.size)
    z1 = rng.standard_normal(qq.size)

    nQ = a * z0
    nU = b * z0 + c * z1

    return nQ.reshape(shape), nU.reshape(shape)