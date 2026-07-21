"""Regularized fixed-layer inversion; optional SPT prior (joint inversion)."""
import numpy as np
from scipy.optimize import minimize

from .forward import forward_vph


def resolution_depth(wlc, P):
    return wlc.max() / P["resolution_divisor"]


def invert(wlc, vc, sigma, P, priors=None, w_p=0.0, x0=None, zbot=None,
           seed=42):
    """Minimize  chi_d^2 + w_p*chi_p^2 + rough_weight*mean(d ln Vs)^2.

    priors: [(z_mid_m, vs_prior_ms, sigma_ln)] or None.
    Returns (thick, vs, rms) with fixed fine_h layers over a half-space.
    """
    f_obs = vc / wlc
    if zbot is None:
        zbot = resolution_depth(wlc, P)
        if priors:
            zbot = max(zbot, max(z for z, _, _ in priors) + 0.5)
    n_lay = max(3, int(round(zbot / P["fine_h"])))
    thick = np.full(n_lay, P["fine_h"])
    zmid = np.concatenate([np.cumsum(thick) - P["fine_h"] / 2,
                           [zbot + 1.0]])
    n = n_lay + 1

    pv = np.full(n, np.nan)
    ps = np.full(n, np.nan)
    for z, v, s in (priors or []):
        k = min(int(z / P["fine_h"]), n - 1)
        pv[k], ps[k] = v, s
    pmask = np.isfinite(pv)

    lo, hi = np.log(P["vs_min"]), np.log(P["vs_max"])

    def objective(logv):
        logv = np.clip(logv, lo, hi)
        vs = np.exp(logv)
        try:
            pred = forward_vph(thick, vs, f_obs, P)
        except Exception:
            return 1e6
        chi_d = np.mean(((pred - vc) / sigma) ** 2)
        chi_p = np.mean(((logv[pmask] - np.log(pv[pmask])) /
                         ps[pmask]) ** 2) if pmask.any() else 0.0
        rough = np.mean(np.diff(logv) ** 2)
        return chi_d + w_p * chi_p + P["rough_weight"] * rough

    rng = np.random.default_rng(seed)
    starts = []
    if x0 is not None and len(x0) == n:
        starts.append(np.log(x0))
    if priors:
        zp = [p[0] for p in priors]
        vp = [p[1] for p in priors]
        vstart = np.interp(zmid, [0.0] + zp + [zbot + 2],
                           [vc.min() / 0.93] + vp + [vp[-1]])
    else:
        vstart = np.interp(zmid, [0.0, zbot + 2],
                           [vc.min() / 0.93, vc.max() / 0.93])
    starts.append(np.log(vstart))
    starts.append(np.log(vstart) + rng.normal(0, 0.1, n))

    best = None
    for s0 in starts:
        res = minimize(objective, s0, method="Nelder-Mead",
                       options={"maxiter": 6000, "xatol": 1e-4,
                                "fatol": 1e-7, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    vs = np.clip(np.exp(best.x), P["vs_min"], P["vs_max"])
    pred = forward_vph(thick, vs, f_obs, P)
    rms = np.sqrt(np.mean(((pred - vc) / vc) ** 2))
    return thick, vs, rms


def profile_at(thick, vs, z):
    zi = np.cumsum(thick)
    k = np.searchsorted(zi, z)
    return vs[min(k, len(vs) - 1)]


def joint_inversion_sweep(wlc, vc, sigma, P, priors, reference,
                          log=print):
    """Increase the prior weight until R^2 (about the 1:1 line, against
    `reference` = [(z, vs_pred)]) reaches target_r2. Returns the final
    (thick, vs, rms, w_p, r2, history)."""
    zbot = max(resolution_depth(wlc, P),
               max(z for z, _, _ in priors) + 0.5)
    x0, result, history = None, None, []
    for w_p in P["wp_sweep"]:
        thick, vs, rms = invert(wlc, vc, sigma, P, priors=priors, w_p=w_p,
                                x0=x0, zbot=zbot)
        x0 = vs
        x = np.array([v for _, v in reference])
        y = np.array([profile_at(thick, vs, z) for z, _ in reference])
        r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
        history.append((w_p, r2, rms))
        log(f"    w_p={w_p:5.1f}: R2={r2:6.2f}  dispersion RMS={rms*100:.1f}%")
        result = (thick, vs, rms, w_p, r2)
        if r2 >= P["target_r2"]:
            break
    return (*result, history)
