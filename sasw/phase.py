"""Phase processing and dispersion extraction (single record)."""
import numpy as np
from scipy.signal import savgol_filter


def process_phase(freq, real, imag):
    """Unwrap the cross-power phase, fix the sign convention, smooth."""
    phase = np.unwrap(np.arctan2(imag, real))
    if np.mean(phase[freq > 10]) > 0:
        phase = -phase
    if phase[len(phase) // 2] < 0:
        phase = -phase
    if len(phase) > 15:
        phase = savgol_filter(phase, 11, 3)
    return phase


def _segments(good, min_len):
    """Index ranges [i0, i1] of consecutive True samples, length >= min_len."""
    segs, start = [], None
    for i, g in enumerate(np.append(good, False)):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= min_len:
                segs.append((start, i - 1))
            start = None
    return segs


def process_phase_segmented(freq, real, imag, coh, spacing, P):
    """Segment-wise unwrapping joined by group-delay continuity.

    np.unwrap walks through every coherence notch inside the band; noise in
    a notch can add or drop one 2*pi cycle, and the slip is then carried to
    every higher frequency (the forward and reverse records slip at
    different notches, which is why they disagree by ~1 cycle at short
    wavelengths). Here the wrapped phase is unwrapped only inside coherent
    segments. The first segment keeps the legacy cycle count; each later
    segment is shifted by the integer number of cycles that best continues
    the previous segment, extrapolated across the gap with its local slope
    d(phi)/df (the group delay, which is smooth for surface waves).
    """
    legacy = process_phase(freq, real, imag)          # for sign + anchor
    raw = np.unwrap(np.arctan2(imag, real))
    sgn = 1.0 if np.corrcoef(raw, legacy)[0, 1] >= 0 else -1.0
    w = sgn * np.arctan2(imag, real)
    ref = sgn * raw
    thr = coherence_threshold(spacing, P) * P.get("seg_coh_factor", 1.0)
    segs = _segments((coh >= thr) & (freq > 0), P.get("seg_min_len", 4))
    if not segs:
        return legacy
    phase = np.full(len(freq), np.nan)
    prev = None
    for i0, i1 in segs:
        seg = np.unwrap(w[i0:i1 + 1])
        if prev is None:
            n = np.round((ref[i0] - seg[0]) / (2 * np.pi))
        else:
            j0, j1 = prev
            k = max(j0, j1 - P.get("seg_slope_pts", 8))
            ff, pp = freq[k:j1 + 1], phase[k:j1 + 1]
            slope = np.polyfit(ff, pp, 1)[0] if len(ff) >= 3 else 0.0
            slope = max(slope, 0.0)          # phase grows with frequency
            pred = phase[j1] + slope * (freq[i0] - freq[j1])
            n = np.round((pred - seg[0]) / (2 * np.pi))
        phase[i0:i1 + 1] = seg + 2 * np.pi * n
        prev = (i0, i1)
    # fill the notches (masked out later by the coherence test anyway)
    ok = np.isfinite(phase)
    phase = np.interp(freq, freq[ok], phase[ok])
    if len(phase) > 15:
        phase = savgol_filter(phase, 11, 3)
    return phase


def coherence_threshold(spacing, P):
    return (P["coh_threshold_short"] if spacing <= P["coh_spacing_break"]
            else P["coh_threshold_long"])


def phase_to_dispersion(freq, phase, coh, spacing, P):
    """Masked dispersion points (wavelength, Vph, freq) from unwrapped phase.

    Masking: coherence threshold, phase floor, positive local phase slope
    (a propagating wave's unwrapped phase must increase with frequency),
    wavelength window lambda in [d*lmin_ratio, d*lmax_ratio], Vph limits.
    """
    dphi = np.gradient(phase, freq)
    pos = dphi > 0
    slope_ok = (dphi > 0.2 * np.median(dphi[pos])) if pos.any() else \
        np.zeros(len(phase), bool)
    mask = ((freq > 0) & (coh >= coherence_threshold(spacing, P)) &
            (phase > 0.1) & slope_ok)
    f, phi = freq[mask], phase[mask]
    if len(f) == 0:
        return np.array([]), np.array([]), np.array([])
    wl = 2 * np.pi * spacing / phi
    vph = f * wl
    ok = ((wl >= P["lambda_min_ratio"] * spacing) &
          (wl <= P["lambda_max_ratio"] * spacing) &
          (vph > P["vph_min"]) & (vph < P["vph_max"]))
    return wl[ok], vph[ok], f[ok]


def irf_gate(freq, real, imag, coh, spacing, P):
    """Impulse-response filtering of a transfer function (cf. Joh 1996).

    The transfer function between the two receivers is transformed to the
    time domain. A surface wave of velocity V crosses the spacing d in
    d/V, so all admissible energy arrives between zero lag and
    d / irf_vmin. Energy outside that gate - late reflections and
    scattered arrivals, ambient noise (which is spread over the whole
    record) and anything at negative lag - is removed with a
    cosine-tapered gate, and the gated response is transformed back.
    Before the transform each frequency bin is weighted by its coherence,
    so that incoherent bins, where the ratio of two noise spectra can be
    large, do not leak into their neighbours through the gate.

    Returns (real, imag) of the filtered transfer function on the same
    frequency grid; the propagation direction is set to positive lag."""
    H = (np.asarray(real) + 1j * np.asarray(imag)) * np.clip(coh, 0.0, 1.0)
    H = np.where(np.isfinite(H), H, 0.0)
    N = len(freq)
    df = freq[1] - freq[0]
    n = 2 * (N - 1)
    h = np.fft.irfft(H, n=n)
    dt = 1.0 / (n * df)
    lag = np.arange(n) * dt
    lag[lag > n * dt / 2] -= n * dt                 # signed lag
    tmax = spacing / P.get("irf_vmin", 60.0)
    tap = max(P.get("irf_taper_s", 0.010), P.get("irf_taper_frac", 0.25) * tmax)
    pos = np.sum(h[(lag >= 0) & (lag <= tmax)] ** 2)
    neg = np.sum(h[(lag < 0) & (lag >= -tmax)] ** 2)
    if neg > pos:                                    # wave travels Y -> X
        H = np.conj(H)
        h = np.fft.irfft(H, n=n)
    lead = P.get("irf_lead_s", 0.004)
    w = np.zeros(n)
    w[(lag >= -lead) & (lag <= tmax)] = 1.0
    m = (lag > tmax) & (lag <= tmax + tap)
    w[m] = 0.5 * (1 + np.cos(np.pi * (lag[m] - tmax) / tap))
    m = (lag < -lead) & (lag >= -lead - tap)
    w[m] = 0.5 * (1 + np.cos(np.pi * (-lead - lag[m]) / tap))
    Hf = np.fft.rfft(h * w)
    return Hf.real, Hf.imag
