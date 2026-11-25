"""Simulator for the neural fusion estimator.

A random ground profile (ln Vs on fixed graded layers) is passed through the
same physics as the field data: fundamental-mode Rayleigh dispersion
(Thomson-Haskell, disba), sampled on a fixed log-wavelength grid. The noise,
the wavelength coverage and the synthetic KPT sounding are added later, on the
GPU, so that each stored profile yields unlimited training examples.

Output (npz): m (N, L) ln Vs per layer, lnv (N, G) ln phase velocity on the
wavelength grid (NaN where the grid lies outside the computed band).
usage: python sim.py N out.npz [workers] [seed]"""
import sys, time, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")

FINE_H, H_GROWTH, H_MAX, ZBOT = 0.4, 1.15, 2.0, 24.0
POISSON, RHO = 0.35, 1.85
LAM = np.geomspace(0.4, 60.0, 32)              # wavelength grid (m)
FREQ = np.geomspace(1.0, 600.0, 72)            # frequencies solved (Hz)


def layers():
    h, z, out = FINE_H, 0.0, []
    while z + 1e-9 < ZBOT:
        out.append(h); z += h; h = min(h * H_GROWTH, H_MAX)
    return np.array(out)


THICK = layers()
ZTOP = np.concatenate([[0], np.cumsum(THICK)])
ZMID = np.concatenate([ZTOP[:-1] + THICK / 2, [ZTOP[-1] + 1.0]])
L = len(THICK) + 1


def sample_profile(rng):
    """Prior over profiles: a stiffening trend, a correlated fluctuation and
    optional features (a stiff stratum, a softer layer, a stiff crust)."""
    z = ZMID
    m = rng.normal(np.log(160.0), 0.38)                       # surface level
    a, z0 = rng.uniform(0.0, 0.55), np.exp(rng.uniform(np.log(0.5), np.log(6.0)))
    prof = m + a * np.log1p(z / z0)
    sd, ell = rng.uniform(0.03, 0.30), np.exp(rng.uniform(np.log(0.6), np.log(6.0)))
    K = sd ** 2 * np.exp(-0.5 * ((z[:, None] - z[None, :]) / ell) ** 2) + 1e-7 * np.eye(len(z))
    prof = prof + np.linalg.cholesky(K) @ rng.normal(size=len(z))
    if rng.random() < 0.5:                                    # stiff stratum
        zj, dj, wj = rng.uniform(0.8, 14.0), rng.uniform(0.2, 1.0), rng.uniform(0.2, 2.0)
        prof = prof + dj / (1 + np.exp(-(z - zj) / wj))
    if rng.random() < 0.2:                                    # soft layer
        zc, wc, dc = rng.uniform(0.8, 8.0), rng.uniform(0.4, 2.0), rng.uniform(0.15, 0.6)
        prof = prof - dc * np.exp(-0.5 * ((z - zc) / wc) ** 2)
    if rng.random() < 0.15:                                   # stiff crust
        prof = prof + rng.uniform(0.2, 0.8) * np.exp(-z / rng.uniform(0.3, 1.2))
    return np.clip(prof, np.log(55.0), np.log(1150.0))


def forward(m):
    from disba import PhaseDispersion
    vs = np.exp(m) / 1000.0
    vp = vs * np.sqrt(2 * (1 - POISSON) / (1 - 2 * POISSON))
    th = np.append(THICK, 10.0) / 1000.0
    pd = PhaseDispersion(th, vp, vs, np.full(len(vs), RHO), dc=0.002)
    cp = pd(np.sort(1.0 / FREQ), mode=0, wave="rayleigh")
    f = 1.0 / cp.period
    v = cp.velocity * 1000.0
    lam = v / f
    o = np.argsort(lam)
    out = np.full(len(LAM), np.nan)
    ok = (LAM >= lam[o][0]) & (LAM <= lam[o][-1])
    out[ok] = np.interp(np.log(LAM[ok]), np.log(lam[o]), np.log(v[o]))
    return out


def work(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    M, V = [], []
    while len(M) < n:
        m = sample_profile(rng)
        try:
            v = forward(m)
        except Exception:
            continue
        if np.isfinite(v).sum() >= 20:
            M.append(m); V.append(v)
    return np.array(M, np.float32), np.array(V, np.float32)


if __name__ == "__main__":
    N = int(sys.argv[1]); out = sys.argv[2]
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    chunk = max(100, N // (W * 20))
    jobs = [(seed * 1000003 + i, chunk) for i in range(int(np.ceil(N / chunk)))]
    t0 = time.time(); M, V = [], []
    with Pool(W) as p:
        for i, (m, v) in enumerate(p.imap_unordered(work, jobs), 1):
            M.append(m); V.append(v)
            if i % max(1, len(jobs) // 10) == 0:
                print(f"{i}/{len(jobs)} chunks, {time.time()-t0:.0f} s", flush=True)
    M = np.concatenate(M)[:N]; V = np.concatenate(V)[:N]
    np.savez_compressed(out, m=M, lnv=V, lam=LAM, thick=THICK, zmid=ZMID)
    print("saved", out, M.shape, V.shape, f"{time.time()-t0:.0f} s")
    print("prior Vs percentiles 5/50/95 at layers 0, 3, 10, last:",
          [np.round(np.exp(np.percentile(M[:, k], [5, 50, 95]))).tolist() for k in (0, 3, 10, M.shape[1] - 1)])
    print("finite grid points per sample: median", int(np.median(np.isfinite(V).sum(1))), "of", V.shape[1])
