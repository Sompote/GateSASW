"""Fundamental-mode Rayleigh forward model (Thomson-Haskell via disba)."""
import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError


def forward_vph(thick_m, vs_ms, f_obs, P):
    """Predict fundamental-mode Rayleigh phase velocity (m/s) at the
    observation frequencies for a layered model (last vs = half-space)."""
    vs = np.asarray(vs_ms) / 1000.0
    nu = P["poisson"]
    vp = vs * np.sqrt(2 * (1 - nu) / (1 - 2 * nu))
    thick = np.append(np.asarray(thick_m), 10.0) / 1000.0
    rho = np.full(len(vs), P["rho"])
    periods = np.sort(1.0 / np.asarray(f_obs))
    pd = PhaseDispersion(thick, vp, vs, rho, dc=0.002)
    cp = pd(periods, mode=0, wave="rayleigh")
    if len(cp.period) < 0.7 * len(periods):
        raise DispersionError("root loss")
    return np.interp(1.0 / np.asarray(f_obs), cp.period,
                     cp.velocity * 1000.0)
