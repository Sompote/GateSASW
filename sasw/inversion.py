"""Regularized fixed-layer inversion; optional SPT prior (joint inversion)."""
import numpy as np
from scipy.optimize import minimize

from .forward import forward_vph


def resolution_depth(wlc, P):
    return wlc.max() / P["resolution_divisor"]


def resolution_window(wlc, P):
    """(z_min, z_res): the depth range the dispersion data constrain.
    z_res = lambda_max / resolution_divisor as before; z_min =
    lambda_min / top_divisor (default 3). Foti et al. (2018) give
    lambda_min/3 to lambda_min/2 for the top of the resolved range and
    lambda_max/3 to lambda_max/2 for the bottom. Above z_min the profile
    is an upward extrapolation of the shortest wavelength measured."""
    return (wlc.min() / P.get("top_divisor", 3.0), resolution_depth(wlc, P))


def graded_layers(zbot, P):
    """Layer thicknesses growing geometrically from fine_h to h_max."""
    h, z, out = P["fine_h"], 0.0, []
    while z + 1e-9 < zbot:
        out.append(h)
        z += h
        h = min(h * P.get("h_growth", 1.15), P.get("h_max", 2.0))
    return np.array(out)


def invert(wlc, vc, sigma, P, priors=None, w_p=0.0, x0=None, zbot=None,
           seed=42):
    if P.get("inv_mode", "legacy") == "graded" and not priors:
        return invert_graded(wlc, vc, sigma, P, seed=seed)
    return invert_legacy(wlc, vc, sigma, P, priors, w_p, x0, zbot, seed)


def invert_graded(wlc, vc, sigma, P, seed=42):
    """Revised inversion: depth-graded layers down to lambda_max/model_div
    (so the half-space is not pinned at z_res), log-Vs parameters,
    L-BFGS-B with finite-difference gradients, multi-start.

    Penalty: chi_d^2 + rough_weight * mean(d ln Vs)^2 (thickness-normalised
    to the 0.4 m reference so the smoothing strength does not depend on the
    layer count)."""
    f_obs = vc / wlc
    zbot = max(wlc.max() / P.get("model_div", 2.0), 3 * P["fine_h"])
    thick = graded_layers(zbot, P)
    zmid = np.concatenate([np.cumsum(thick) - thick / 2,
                           [thick.sum() + 1.0]])
    hc = np.concatenate([(thick[:-1] + thick[1:]) / 2,
                         [thick[-1]]])          # centre spacing to next
    n = len(thick) + 1
    lo, hi = np.log(P["vs_min"]), np.log(P["vs_max"])
    rw = P["rough_weight"]

    def objective(logv):
        vs = np.exp(np.clip(logv, lo, hi))
        try:
            pred = forward_vph(thick, vs, f_obs, P)
        except Exception:
            return 1e3 + np.sum((logv - np.log(vc.mean() / 0.93)) ** 2)
        chi_d = np.mean(((pred - vc) / sigma) ** 2)
        g = np.diff(logv) * (P["fine_h"] / hc)   # per 0.4 m equivalent
        return chi_d + rw * np.mean(g ** 2)

    # starts: dispersion-curve mapped profile (lambda/2.5 -> Vph/0.92),
    # a constant profile, and a perturbed copy
    order = np.argsort(wlc)
    vmap = np.interp(zmid, wlc[order] / 2.5, vc[order] / 0.92)
    starts = [np.log(vmap),
              np.full(n, np.log(np.median(vc) / 0.92)),
              np.log(vmap) + np.random.default_rng(seed).normal(0, 0.1, n)]
    best = None
    for s0 in starts:
        res = minimize(objective, np.clip(s0, lo, hi), method="L-BFGS-B",
                       bounds=[(lo, hi)] * n,
                       options={"maxiter": 400, "eps": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    vs = np.clip(np.exp(best.x), P["vs_min"], P["vs_max"])
    pred = forward_vph(thick, vs, f_obs, P)
    rms = np.sqrt(np.mean(((pred - vc) / vc) ** 2))
    return thick, vs, rms


def invert_legacy(wlc, vc, sigma, P, priors=None, w_p=0.0, x0=None,
                  zbot=None, seed=42):
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


def invert_effective(wlc, vc, sigma, dsp, P, log=None):
    """Mode-corrected inversion (fixed point).

    The measured two-receiver velocity is an apparent, mixed-mode quantity
    (sasw.effective). Starting from the fundamental-mode solution, each
    pass computes for the current model the ratio R = V_app / V_1 at every
    composite point (receivers at d and 2d, d = the spacing that produced
    that wavelength), converts the data to fundamental-mode equivalents
    vc / R, and inverts again. The ratio is clipped to [r_min, r_max] so
    that one poorly resolved mode transition cannot dominate a pass.
    Returns (thick, vs, rms_apparent, history)."""
    from .effective import apparent_vph
    f_obs = vc / wlc
    thick, vs, rms = invert(wlc, vc, sigma, P)
    hist = [(0, rms, 1.0)]
    Rprev = np.ones(len(vc))
    for it in range(1, P.get("eff_iters", 3) + 1):
        va, v1 = apparent_vph(thick, vs, f_obs, dsp, P,
                              n_modes=P.get("eff_modes", 4))
        R = np.where(np.isfinite(va / v1), va / v1, 1.0)
        R = np.clip(R, P.get("eff_rmin", 0.7), P.get("eff_rmax", 1.6))
        R = 0.5 * (R + Rprev) if it > 1 else R     # damped update
        thick, vs, _ = invert(wlc * 1.0, vc / R, sigma / R, P)
        change = float(np.max(np.abs(R / Rprev - 1)))
        Rprev = R
        hist.append((it, None, float(np.median(R))))
        if change < 0.02:
            break
    va, _ = apparent_vph(thick, vs, f_obs, dsp, P,
                         n_modes=P.get("eff_modes", 4))
    ok = np.isfinite(va)
    rms = float(np.sqrt(np.mean(((va[ok] - vc[ok]) / vc[ok]) ** 2)))
    return thick, vs, rms, hist, Rprev


def invert_joint(wlc, vc, sigma, P, priors, w_p, zbot=None, x0=None):
    """Penetration-constrained inversion on graded layers.

    Minimises  chi_d^2 + w_p * chi_p^2 + rough_weight * roughness, where
    chi_p^2 = mean_k ((ln Vs_k - ln Vs_prior,k) / s_k)^2 over the layers
    that contain prior points [(z_mid, vs_prior, sigma_ln)] (several
    points in one layer are averaged in log space).

    The result is NOT an independent profile: as w_p grows the model is
    pulled onto the prior wherever the dispersion data allow it. Its use
    is (i) as a compatibility test - can the dispersion data be fitted
    within their uncertainty by a profile that matches the penetration
    data? - and (ii) as a penetration-informed engineering profile.
    Returns (thick, vs, rms_rel, chi_d2, chi_p2)."""
    f_obs = vc / wlc
    if zbot is None:
        zbot = max(wlc.max() / P.get("model_div", 2.0),
                   max(z for z, _, _ in priors) + 0.4)
    thick = graded_layers(zbot, P)
    ztop = np.concatenate([[0], np.cumsum(thick)])
    hc = np.concatenate([(thick[:-1] + thick[1:]) / 2, [thick[-1]]])
    n = len(thick) + 1
    lo, hi = np.log(P["vs_min"]), np.log(P["vs_max"])
    pv = np.full(n, np.nan)
    ps = np.full(n, np.nan)
    for k in range(len(thick)):
        m = [(v, s) for z, v, s in priors if ztop[k] <= z < ztop[k + 1]]
        if m:
            pv[k] = np.mean([np.log(v) for v, _ in m])
            ps[k] = np.mean([s for _, s in m]) / np.sqrt(len(m))
    pm = np.isfinite(pv)

    def parts(logv):
        vs = np.exp(np.clip(logv, lo, hi))
        pred = forward_vph(thick, vs, f_obs, P)
        chi_d = np.mean(((pred - vc) / sigma) ** 2)
        chi_p = np.mean(((logv[pm] - pv[pm]) / ps[pm]) ** 2) if pm.any() \
            else 0.0
        g = np.diff(logv) * (P["fine_h"] / hc)
        return chi_d, chi_p, np.mean(g ** 2), pred

    def objective(logv):
        try:
            cd, cp, rg, _ = parts(logv)
        except Exception:
            return 1e3 + np.sum((logv - np.log(vc.mean() / 0.93)) ** 2)
        return cd + w_p * cp + P["rough_weight"] * rg

    order = np.argsort(wlc)
    zmid = np.concatenate([np.cumsum(thick) - thick / 2, [thick.sum() + 1]])
    vmap = np.log(np.interp(zmid, wlc[order] / 2.5, vc[order] / 0.92))
    vpri = vmap.copy()
    vpri[pm] = pv[pm]
    starts = [vpri, vmap]
    if x0 is not None and len(x0) == n:
        starts.insert(0, np.log(x0))
    best = None
    for s0 in starts:
        res = minimize(objective, np.clip(s0, lo, hi), method="L-BFGS-B",
                       bounds=[(lo, hi)] * n,
                       options={"maxiter": 400, "eps": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    cd, cp, _, pred = parts(best.x)
    vs = np.exp(np.clip(best.x, lo, hi))
    rms = float(np.sqrt(np.mean(((pred - vc) / vc) ** 2)))
    return thick, vs, rms, float(cd), float(cp)
