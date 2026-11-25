"""Bayesian fusion of a surface-wave dispersion curve with a penetration
sounding at the same foundation.

Unknown: m = ln Vs of the graded layers (and half-space).

Surface-wave likelihood (an area test, taken as unbiased in level):
    d_i = F_i(m) + e_i,            e_i ~ N(0, sigma_i^2)
with F the fundamental-mode forward model applied to the mode-corrected
composite curve.

Penetration likelihood (a point test). The velocity converted from the
blow counts in layer k is the true layer velocity plus a *site offset* b
shared by the whole sounding (conversion error, the particular location of
the hole) plus increment noise:
    y_k = m_k + b + e_k,   b ~ N(0, tau^2),   e_k ~ N(0, sigma_w^2 / n_k)
Marginalising b gives y ~ N(A m, S),  S = diag(sigma_w^2/n_k) + tau^2 11^T,
so the sounding informs the *shape* of the profile strongly and its
*level* only weakly. tau and sigma_w are learned from all soundings.

Smoothness prior: lambda_r * sum (h0/hbar_l * (m_{l+1}-m_l))^2, with
lambda_r = n * w_r / (L-1) so that it equals the roughness term of the
standard inversion.

Outputs: MAP profile, Laplace posterior standard deviation of ln Vs per
layer, the estimated site offset, and a conflict statistic (chi-square of
the sounding under the surface-wave-only posterior predictive).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .forward import forward_vph
from .inversion import graded_layers


def _setup(wlc, vc, sigma, P, zbot):
    f_obs = vc / wlc
    thick = graded_layers(zbot, P)
    hc = np.concatenate([(thick[:-1] + thick[1:]) / 2, [thick[-1]]])
    n = len(thick) + 1
    D = np.zeros((n - 1, n))
    for l in range(n - 1):
        D[l, l], D[l, l + 1] = -P["fine_h"] / hc[l], P["fine_h"] / hc[l]
    lam_r = len(vc) * P["rough_weight"] / (n - 1)
    return f_obs, thick, D, lam_r, n


def _forward(m, thick, f_obs, P):
    return forward_vph(thick, np.exp(m), f_obs, P)


def _jac(m, thick, f_obs, sigma, P, eps=2e-3):
    """Jacobian of the normalised prediction F/sigma with respect to m."""
    f0 = _forward(m, thick, f_obs, P)
    J = np.zeros((len(f0), len(m)))
    for k in range(len(m)):
        mp = m.copy()
        mp[k] += eps
        try:
            J[:, k] = (_forward(mp, thick, f_obs, P) - f0) / eps
        except Exception:
            J[:, k] = 0.0
    return J / sigma[:, None], f0


def kpt_design(thick, z_mid, lnv, n_layers):
    """Layer-averaged sounding: returns (A, y, n_k) for layers that contain
    admissible increments."""
    ztop = np.concatenate([[0], np.cumsum(thick)])
    rows, y, nk = [], [], []
    for k in range(len(thick)):
        m = (z_mid >= ztop[k]) & (z_mid < ztop[k + 1])
        if m.any():
            a = np.zeros(n_layers)
            a[k] = 1.0
            rows.append(a)
            y.append(float(np.mean(lnv[m])))
            nk.append(int(m.sum()))
    return np.array(rows), np.array(y), np.array(nk)


def kpt_cov(nk, tau, sigma_w):
    return np.diag(sigma_w ** 2 / nk) + tau ** 2 * np.ones((len(nk), len(nk)))


def fit(wlc, vc, sigma, P, kpt=None, tau=0.28, sigma_w=0.19, zbot=None,
        x0=None):
    """MAP + Laplace. kpt = (z_mid array, ln Vs_KPT array) or None for the
    surface-wave-only posterior. Returns a dict."""
    if zbot is None:
        zbot = max(wlc.max() / P.get("model_div", 2.0), 3 * P["fine_h"])
        if kpt is not None and len(kpt[0]):
            zbot = max(zbot, float(np.max(kpt[0])) + P["fine_h"])
    f_obs, thick, D, lam_r, n = _setup(wlc, vc, sigma, P, zbot)
    lo, hi = np.log(P["vs_min"]), np.log(P["vs_max"])
    A = y = Sinv = None
    if kpt is not None and len(kpt[0]):
        A, y, nk = kpt_design(thick, np.asarray(kpt[0]), np.asarray(kpt[1]), n)
        if len(y):
            Sinv = np.linalg.inv(kpt_cov(nk, tau, sigma_w))
        else:
            A = None

    def obj(m):
        m = np.clip(m, lo, hi)
        try:
            r = (_forward(m, thick, f_obs, P) - vc) / sigma
        except Exception:
            return 1e6 + np.sum((m - np.log(vc.mean() / 0.93)) ** 2)
        g = D @ m
        val = np.sum(r ** 2) + lam_r * np.sum(g ** 2)
        if A is not None:
            e = y - A @ m
            val += e @ Sinv @ e
        return 0.5 * val

    order = np.argsort(wlc)
    zmid = np.concatenate([np.cumsum(thick) - thick / 2, [thick.sum() + 1]])
    starts = [np.log(np.interp(zmid, wlc[order] / 2.5, vc[order] / 0.92))]
    if x0 is not None and len(x0) == n:
        starts.insert(0, np.asarray(x0, float))
    best = None
    for s0 in starts:
        res = minimize(obj, np.clip(s0, lo, hi), method="L-BFGS-B",
                       bounds=[(lo, hi)] * n,
                       options={"maxiter": 500, "eps": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    m = np.clip(best.x, lo, hi)
    J, f0 = _jac(m, thick, f_obs, sigma, P)
    H = J.T @ J + lam_r * D.T @ D
    if A is not None:
        H = H + A.T @ Sinv @ A
    C = np.linalg.pinv(H + 1e-9 * np.eye(n))
    r = (f0 - vc) / sigma
    out = dict(thick=thick, m=m, vs=np.exp(m), sd=np.sqrt(np.clip(np.diag(C), 0, None)),
               cov=C, chi_d=float(np.mean(r ** 2)),
               rms=float(np.sqrt(np.mean(((f0 - vc) / vc) ** 2))),
               A=A, y=y)
    if A is not None:
        e = y - A @ m
        one = np.ones(len(y))
        out["offset"] = float(tau ** 2 * one @ Sinv @ e)
        out["nk"] = nk
    return out


def predictive(sw, A, y, nk, tau, sigma_w):
    """Log predictive density and chi-square of a sounding under the
    surface-wave-only posterior `sw` (a dict from fit(..., kpt=None))."""
    S = A @ sw["cov"] @ A.T + kpt_cov(nk, tau, sigma_w)
    e = y - A @ sw["m"]
    Sinv = np.linalg.inv(S)
    q = float(e @ Sinv @ e)
    _, logdet = np.linalg.slogdet(S)
    k = len(y)
    return dict(logp=-0.5 * (q + logdet + k * np.log(2 * np.pi)), chi2=q,
                dof=k, p=float(chi2.sf(q, k)))
