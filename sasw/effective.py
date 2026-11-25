"""Apparent (effective) two-receiver phase velocity from modal superposition.

A vertical point source at r = 0 produces, at the free surface, the
far-field vertical displacement (Lai 1998; Lai & Rix 1998; Foti et al. 2014)

    u_z(r, w) ~ sum_j A_j H0^(2)(k_j r),   A_j = r_z,j(0)^2 / (V_j U_j I_j),

with V_j, U_j the modal phase and group velocities and
I_j = 1/2 int rho (r_x^2 + r_z^2) dz the energy integral (the ratio
r_z(0)^2 / I_j does not depend on the eigenfunction normalisation).
The SASW test measures the phase difference between receivers at r1 and r2,
so the apparent velocity is

    V_SASW(w; r1, r2) = w (r2 - r1) / [psi(r2) - psi(r1)],
    psi(r) = -arg u_z(r, w),

which equals the fundamental-mode velocity only when one mode dominates.
In the common-receiver-midpoint layout the near receiver is one spacing
from the source: r1 = d, r2 = 2d. Fitting V_SASW as if it were the
fundamental mode overestimates Vs where higher modes contribute
(Foti et al. 2018, Sect. 3.3).
"""
import numpy as np
from scipy.special import hankel2
from disba import PhaseDispersion, GroupDispersion, EigenFunction


def _model(thick, vs_ms, P):
    vs = np.asarray(vs_ms) / 1000.0
    nu = P["poisson"]
    vp = vs * np.sqrt(2 * (1 - nu) / (1 - 2 * nu))
    th = np.append(np.asarray(thick), 10.0) / 1000.0
    rho = np.full(len(vs), P["rho"])
    return th, vp, vs, rho


def _fine(th, vp, vs, rho, lam_km, nsub=4, depth_mult=3.0):
    """Split layers into sub-layers (for the energy integral) and extend the
    half-space as thin layers down to depth_mult * wavelength."""
    T, A, B, R = [], [], [], []
    for h, a, b, r in zip(th[:-1], vp[:-1], vs[:-1], rho[:-1]):
        n = max(1, int(np.ceil(h / (lam_km / (6 * nsub)))))
        n = min(n, 12)
        T += [h / n] * n; A += [a] * n; B += [b] * n; R += [r] * n
    ztop = sum(T)
    zmax = max(ztop + depth_mult * lam_km, ztop + 0.002)
    dz = lam_km / 12
    while ztop < zmax:
        T.append(dz); A.append(vp[-1]); B.append(vs[-1]); R.append(rho[-1])
        ztop += dz
    T.append(1.0); A.append(vp[-1]); B.append(vs[-1]); R.append(rho[-1])
    return (np.array(T), np.array(A), np.array(B), np.array(R))


def modal_weights(thick, vs_ms, f, P, n_modes=4):
    """[(k_j [1/m], A_j)] for the propagating Rayleigh modes at frequency f."""
    th, vp, vs, rho = _model(thick, vs_ms, P)
    T = 1.0 / f
    pd = PhaseDispersion(th, vp, vs, rho, dc=0.001)
    gd = GroupDispersion(th, vp, vs, rho, dc=0.001)
    out = []
    for m in range(n_modes):
        try:
            c = pd(np.array([T]), mode=m, wave="rayleigh")
            if len(c.velocity) == 0:
                break
            V = float(c.velocity[0])                       # km/s
            g = gd(np.array([T]), mode=m, wave="rayleigh")
            U = float(g.velocity[0]) if len(g.velocity) else V
            lam = V * T
            Tf, Af, Bf, Rf = _fine(th, vp, vs, rho, lam)
            ef = EigenFunction(Tf, Af, Bf, Rf)(T, mode=m, wave="rayleigh")
            z, ur, uz = ef.depth, ef.ur, ef.uz
            dz = np.diff(z)
            Rl = Rf[:len(dz)]
            e = 0.5 * (ur[:-1] ** 2 + uz[:-1] ** 2 + ur[1:] ** 2 + uz[1:] ** 2)
            I = 0.5 * np.sum(Rl * e * dz)
            if not np.isfinite(I) or I <= 0 or abs(U) < 1e-6:
                continue
            A = uz[0] ** 2 / (V * U * I)
            out.append((2 * np.pi * f / (V * 1000.0), A))
        except Exception:
            break
    return out


def apparent_vph(thick, vs_ms, freqs, spacings, P, n_modes=4, nr=41):
    """Two-receiver apparent phase velocity (m/s) at each frequency, with
    receivers at r1 = d and r2 = 2d from the source (CRMP)."""
    out = np.empty(len(freqs))
    v1 = np.empty(len(freqs))
    for i, (f, d) in enumerate(zip(freqs, spacings)):
        mw = modal_weights(thick, vs_ms, f, P, n_modes)
        if not mw:
            out[i] = v1[i] = np.nan
            continue
        k = np.array([m[0] for m in mw])
        A = np.array([m[1] for m in mw])
        r = np.linspace(d, 2 * d, nr)
        u = (A[None, :] * hankel2(0, k[None, :] * r[:, None])).sum(1)
        psi = np.unwrap(-np.angle(u))
        out[i] = 2 * np.pi * f * d / (psi[-1] - psi[0])
        v1[i] = 2 * np.pi * f / k[0]
    return out, v1


def bin_spacing(wl_raw, sp_raw, wlc):
    """Representative receiver spacing for each composite wavelength: the
    geometric mean of the spacings of the raw points nearest in log-lambda."""
    lw = np.log(wl_raw)
    out = []
    for w in wlc:
        idx = np.argsort(np.abs(lw - np.log(w)))[:max(5, len(lw) // 30)]
        out.append(float(np.exp(np.mean(np.log(sp_raw[idx])))))
    return np.array(out)
