"""Multi-baseline (cross-spacing) phase unwrapping.

A two-receiver phase spectrum is known only modulo 2*pi. np.unwrap fixes
the cycle count by walking along frequency, so one noisy band shifts every
frequency above it by a whole cycle; on the 164-site dataset the forward
and reverse phases of a spacing differ by ~0, ~1 or ~2 cycles (bimodal),
not by a smooth drift. A single record cannot tell which count is right.

The other spacings can. At a given frequency the propagation phase scales
with the receiver spacing, phi_d(f) = 2*pi*f*d / V(f), so a shorter
spacing - which wraps fewer times - predicts the phase of a longer one:

    phi_ref(f; d) = (d / d_prev) * phi_prev(f).

The prediction only has to be right to within half a cycle, i.e. the
apparent velocities of the two spacings may differ by up to lambda/(2d).
Spacings are processed from short to long. Each record is cut into
coherent segments, each segment is unwrapped internally, and its integer
cycle count is the one that best matches the prediction. The shortest
spacing has no predictor and is anchored physically: at the start of its
first coherent segment the phase must lie in [-pi/2, 3*pi/2), which holds
while f*d/V < 0.75; later segments follow by group-delay continuity.
(cf. the wavenumber unwrapping of Joh, Vivar & Akmal 2023.)

Lateral pass (optional, P["chain_prior"] = (freq, velocity)): the anchor is
the weak link, because adding one cycle to the shortest spacing keeps the
whole chain self-consistent. When several sites share a structure, the
median dispersion curve of the *other* sites predicts the anchor phase as
well. The correction is still an integer number of cycles per segment, so
it moves a site off a wrong branch but cannot smooth sites together.
"""
import numpy as np
from scipy.signal import savgol_filter, medfilt

from .io import read_record
from .phase import (coherence_threshold, phase_to_dispersion, _segments,
                    irf_gate)

TWO_PI = 2 * np.pi


def _signed_wrapped(freq, real, imag, coh, thr):
    """Wrapped phase with the sign that makes it grow with frequency."""
    w = np.arctan2(imag, real)
    segs = _segments((coh >= thr) & (freq > 0), 4)
    if not segs:
        return w, segs
    i0, i1 = max(segs, key=lambda s: s[1] - s[0])
    u = np.unwrap(w[i0:i1 + 1])
    if np.polyfit(freq[i0:i1 + 1], u, 1)[0] < 0:
        w = -w
    return w, segs


class _Reference:
    """Velocity-versus-frequency predictor built from resolved spacings."""

    def __init__(self):
        self.parts = []                      # (spacing, f, V) largest last

    def add(self, spacing, f, phase):
        ok = np.isfinite(phase) & (phase > 0.3) & (f > 0)
        if ok.sum() < 6:
            return
        v = TWO_PI * f[ok] * spacing / phase[ok]
        k = 5 if len(v) >= 5 else 1
        self.parts.append((spacing, f[ok], medfilt(v, k) if k > 1 else v))

    def ready(self):
        return bool(self.parts)

    def velocity(self, f):
        """Prefer the largest resolved spacing that covers f; outside all
        coverage hold the velocity at the nearest covered frequency."""
        out = np.full(len(f), np.nan)
        for _, fp, vp in self.parts:         # later (larger) overwrite
            m = (f >= fp.min()) & (f <= fp.max())
            out[m] = np.interp(f[m], fp, vp)
        if np.isnan(out).any():
            fa = np.concatenate([p[1] for p in self.parts])
            va = np.concatenate([p[2] for p in self.parts])
            o = np.argsort(fa)
            miss = np.isnan(out)
            out[miss] = np.interp(f[miss], fa[o], va[o])
        return out


def _resolve(freq, w, segs, spacing, ref, P):
    """Assign integer cycles to every coherent segment. Returns
    (phase with NaN outside segments, worst segment residual in cycles)."""
    phase = np.full(len(freq), np.nan)
    worst, prev = 0.0, None
    vref = ref.velocity(freq) if ref.ready() else None
    for i0, i1 in segs:
        seg = np.unwrap(w[i0:i1 + 1])
        if vref is not None:
            pred = TWO_PI * freq[i0:i1 + 1] * spacing / vref[i0:i1 + 1]
            n = np.round(np.median(pred - seg) / TWO_PI)
            res = np.median(np.abs(seg + TWO_PI * n - pred)) / TWO_PI
            worst = max(worst, float(res))
        elif prev is None:
            # anchor: start phase in [-lo, 1-lo) cycle. Field data: 71 % of
            # anchor phases lie in 0-0.25 cycle, none in 0.75-0.9 and 8 % in
            # 0.9-1.0 (near-zero phase pushed negative by noise).
            lo = P.get("chain_anchor_lo", 0.25)
            n = -np.floor(np.median(seg[:3]) / TWO_PI + lo)
            n += P.get("chain_anchor_shift", 0)      # diagnostic only
        else:
            j0, j1 = prev
            k = max(j0, j1 - 8)
            slope = max(np.polyfit(freq[k:j1 + 1], phase[k:j1 + 1], 1)[0],
                        0.0) if j1 - k >= 2 else 0.0
            pred0 = phase[j1] + slope * (freq[i0] - freq[j1])
            n = np.round((pred0 - seg[0]) / TWO_PI)
        phase[i0:i1 + 1] = seg + TWO_PI * n
        prev = (i0, i1)
    return phase, worst


def _filled(freq, phase):
    ok = np.isfinite(phase)
    out = np.interp(freq, freq[ok], phase[ok])
    if len(out) > 15:
        out = savgol_filter(out, 11, 3)
    return out


def chain_phases(data_dir, tests, P, log=lambda m: None):
    """Resolve the phase of every record of a site, short spacing first.
    Returns {folder: (freq, phase, coh)}; records without a coherent
    segment are omitted."""
    from .qc import record_quality, fr_disagreement
    raw = {}
    for folder, spacing, direction in tests:
        try:
            freq, real, imag, coh = read_record(data_dir, folder)
        except Exception as e:
            log(f"{folder}: read error {e}")
            continue
        if P.get("irf"):
            real, imag = irf_gate(freq, real, imag, coh, spacing, P)
        thr = coherence_threshold(spacing, P)
        w, segs = _signed_wrapped(freq, real, imag, coh, thr)
        if segs:
            raw.setdefault(spacing, []).append(
                (folder, direction, freq, w, coh, segs))
    ref, out = _Reference(), {}
    prior = P.get("chain_prior")
    if prior is not None:
        # lateral predictor: median velocity-frequency curve of the other
        # sites on the same structure (lowest priority; any spacing
        # resolved at this site overrides it where they overlap)
        fp, vp = np.asarray(prior[0], float), np.asarray(prior[1], float)
        o = np.argsort(fp)
        ref.parts.append((0.0, fp[o], vp[o]))
    for spacing in sorted(raw):
        cands = []
        for folder, direction, freq, w, coh, segs in raw[spacing]:
            ph, worst = _resolve(freq, w, segs, spacing, ref, P)
            full = _filled(freq, ph)
            out[folder] = (freq, full, coh)
            wl, vph, f = phase_to_dispersion(freq, full, coh, spacing, P)
            n, rho = record_quality(wl, vph, f, P)
            if worst > P.get("chain_ambiguous", 0.35):
                log(f"{folder}: cycle count ambiguous "
                    f"(residual {worst:.2f} cycle)")
            if P.get("chain_strict") and worst > P.get("chain_ambiguous", 0.35):
                del out[folder]
                log(f"{folder}: REJECTED, ambiguous cycle count")
                continue
            if n >= P["min_pts"] and rho >= P["spearman_min"]:
                cands.append((direction, freq, ph, wl, vph, worst))
        if not cands:
            continue
        # extend the predictor with this spacing: best-agreeing F/R pair,
        # else the single best record
        F = [c for c in cands if c[0] == "F"]
        R = [c for c in cands if c[0] == "R"]
        best = None
        for a in F:
            for b in R:
                dis = fr_disagreement(a[3], a[4], b[3], b[4])
                if best is None or dis < best[0]:
                    best = (dis, a, b)
        if best is not None and best[0] <= P.get("chain_pair_max", 0.5):
            _, a, b = best
            fg = a[1]
            pb = np.interp(fg, b[1], np.where(np.isfinite(b[2]), b[2], np.nan))
            both = np.isfinite(a[2]) & np.isfinite(pb)
            ph = np.where(both, (a[2] + pb) / 2, np.nan)
            ref.add(spacing, fg, ph)
        else:
            c = min(cands, key=lambda c: c[5])
            ref.add(spacing, c[1], c[2])
    return out
