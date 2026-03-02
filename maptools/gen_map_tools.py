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

    Special-cases zero-variance pixels:
      - QQ=UU=QU=0 -> (nQ,nU)=(0,0)
      - QQ=0, UU>0 -> nQ=0, nU~N(0,UU) (requires QU=0)
      - UU=0, QQ>0 -> nU=0, nQ~N(0,QQ) (requires QU=0)
    """
    sig2_QQ = np.asarray(sig2_QQ, dtype=float)
    sig2_UU = np.asarray(sig2_UU, dtype=float)
    sig2_QU = np.asarray(sig2_QU, dtype=float)
    if sig2_QQ.shape != sig2_UU.shape or sig2_QQ.shape != sig2_QU.shape:
        raise ValueError("sig2_QQ, sig2_UU, sig2_QU must have identical shapes")

    if rng is None:
        rng = np.random.default_rng()

    shape = sig2_QQ.shape
    qq = sig2_QQ.reshape(-1) + eps
    uu = sig2_UU.reshape(-1) + eps
    qu = sig2_QU.reshape(-1)

    n = qq.size
    nQ = np.zeros(n, dtype=float)
    nU = np.zeros(n, dtype=float)

    # Masks for degenerate cases (use eps-adjusted qq/uu for stability, but check raw too if desired)
    both0 = (qq == 0.0) & (uu == 0.0) & (qu == 0.0)  # exact zeros after eps
    qq0   = (qq == 0.0) & ~both0
    uu0   = (uu == 0.0) & ~both0

    # If qq==0 or uu==0 but qu!=0, covariance can't be PSD
    # bad_qu_qq0 = qq0 & (qu != 0.0)
    # bad_qu_uu0 = uu0 & (qu != 0.0)
    # if np.any(bad_qu_qq0 | bad_qu_uu0):
    #     bad = np.where(bad_qu_qq0 | bad_qu_uu0)[0][:10]
    #     raise ValueError(
    #         f"Invalid covariance: QU!=0 where QQ==0 or UU==0 in pixels (up to 10): {bad}"
    #     )

    # Handle qq==0: Q fixed at 0, U ~ N(0, uu)
    idx = np.where(qq0 & (uu > 0.0))[0]
    if idx.size:
        nU[idx] = np.sqrt(uu[idx]) * rng.standard_normal(idx.size)

    # Handle uu==0: U fixed at 0, Q ~ N(0, qq)
    idx = np.where(uu0 & (qq > 0.0))[0]
    if idx.size:
        nQ[idx] = np.sqrt(qq[idx]) * rng.standard_normal(idx.size)

    # General case: qq>0 and uu>0
    gen = (~both0) & (qq > 0.0) & (uu > 0.0)

    if np.any(gen):
        qqg = qq[gen]
        uug = uu[gen]
        qug = qu[gen]

        a = np.sqrt(qqg)
        b = np.where(a!=0, qug / a, 0)
        rad = uug - b*b

        # PSD check with tolerance; clamp tiny negatives
        # tol = 1e-12 * np.maximum(uug, 1.0)
        # if np.any(rad < -tol):
        #     bad = np.where(gen)[0][np.where(rad < -tol)[0][:10]]
        #     raise ValueError(
        #         f"Covariance not PSD in pixels (up to 10): {bad}. Try eps>0 or inspect QQ/UU/QU."
        #     )
        rad = np.maximum(rad, 0.0)
        c = np.sqrt(rad)

        z0 = rng.standard_normal(a.size)
        z1 = rng.standard_normal(a.size)

        nQ[gen] = a * z0
        nU[gen] = b * z0 + c * z1

    # If eps made qq/uu exactly 0 unlikely; if eps>0, both0 mask will be false.
    return nQ.reshape(shape), nU.reshape(shape)