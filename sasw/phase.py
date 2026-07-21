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
