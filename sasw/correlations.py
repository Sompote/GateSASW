"""Vs-N empirical correlation pool and site-appropriate selection."""
import numpy as np

# (name, a, b) for Vs = a * N^b  [m/s]
EQUATIONS = [
    ("Kanai (1966)", 19.0, 0.600),
    ("Fujiwara (1972)", 92.1, 0.337),
    ("Imai (1977) all", 91.0, 0.337),
    ("Imai (1977) sand", 80.6, 0.331),
    ("Imai (1977) clay", 102.0, 0.292),
    ("Imai & Tonouchi (1982)", 96.9, 0.314),
    ("Yokota et al. (1981)", 121.0, 0.270),
    ("Athanasopoulos (1995)", 107.6, 0.360),
    ("Iyisan (1996)", 51.5, 0.516),
    ("Kiku et al. (2001)", 68.3, 0.292),
    ("Anbazhagan & Sitharam (2008)", 50.0, 0.410),
    ("Hasancebi & Ulusay (2007) all", 90.0, 0.309),
    ("Hasancebi & Ulusay (2007) sand", 90.82, 0.319),
    ("Hasancebi & Ulusay (2007) clay", 97.89, 0.269),
    ("Maheshwari et al. (2010)", 95.64, 0.301),
    ("Dikmen (2009) all", 58.0, 0.390),
    ("Dikmen (2009) sand", 73.0, 0.330),
    ("Dikmen (2009) clay", 44.0, 0.480),
    ("Fatehnia et al. (2015)", 77.1, 0.355),
]


def select_best(calibration, log=print):
    """Rank equations by RMS log-residual against calibration points
    [(N, vs_sasw)]. Returns (name, a, b, sigma_ln, ranking)."""
    scores = []
    for name, a, b in EQUATIONS:
        res = [np.log(v / (a * N ** b)) for N, v in calibration]
        rms_ln = float(np.sqrt(np.mean(np.square(res))))
        bias = float(np.exp(np.mean(res)))
        scores.append((rms_ln, bias, name, a, b))
    scores.sort()
    log(f"    {'rank':>4} {'equation':32} {'RMS(ln)':>8} {'bias':>6}")
    for r, (s, bias, name, a, b) in enumerate(scores[:5], 1):
        log(f"    {r:>4} {name:32} {s:8.3f} {bias:6.2f}")
    rms_ln, bias, name, a, b = scores[0]
    return name, a, b, rms_ln, scores


def build_priors(spt, name_a_b_sigma, P):
    """Priors [(z_mid, vs_prior, sigma_ln)] from SPT intervals using the
    selected equation."""
    _, a, b, sigma_ln = name_a_b_sigma
    out = []
    for d0, d1, N, status in spt:
        s = max(sigma_ln, P["prior_sigma_min"])
        if status == "refusal":
            s *= P["refusal_sigma_factor"]
        out.append(((d0 + d1) / 2, a * N ** b, s))
    return out
