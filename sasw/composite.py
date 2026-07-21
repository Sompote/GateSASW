"""Composite dispersion curve: median binning, IQR outlier rejection,
trust-weighted uncertainty. No shape constraint is imposed."""
import numpy as np
from scipy.signal import savgol_filter


def build_composite(wl, vph, trust, P):
    bins = np.logspace(np.log10(wl.min() * 0.99), np.log10(wl.max() * 1.01),
                       P["n_bins"] + 1)
    wlc, vc, sg = [], [], []
    for j in range(P["n_bins"]):
        m = (wl >= bins[j]) & (wl < bins[j + 1])
        if m.sum() < 2:
            continue
        v = vph[m]
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        inl = (v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr) if iqr > 0 \
            else np.ones(len(v), bool)
        med = np.median(v[inl])
        wlc.append(np.sqrt(bins[j] * bins[j + 1]))
        vc.append(med)
        sg.append(max(np.std(v[inl]), P["sigma_floor"] * med) *
                  np.mean(trust[m]))
    wlc, vc, sg = map(np.array, (wlc, vc, sg))
    if len(vc) > 7:
        vc = savgol_filter(vc, 7, 2)
    return wlc, vc, sg
