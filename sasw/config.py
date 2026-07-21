"""Default processing parameters. Override any of these via the CLI or by
passing a dict to the pipeline functions."""

DEFAULTS = {
    # masking
    "coh_threshold_short": 0.95,   # spacings <= coh_spacing_break
    "coh_threshold_long": 0.90,    # spacings >  coh_spacing_break
    "coh_spacing_break": 4.0,      # m
    "lambda_min_ratio": 0.5,       # lambda >= d/2
    "lambda_max_ratio": 3.0,       # lambda <= 3d (Heisey et al. 1982)
    "vph_min": 50.0,
    "vph_max": 1500.0,
    # record quality gates
    "min_pts": 10,
    "spearman_min": 0.5,
    "fr_agree": 0.20,              # F/R agreement threshold
    "cross_dev": 0.25,             # cross-spacing validation threshold
    "continuity_step": 0.50,       # unverified-record continuity gate
    "offset_range": 2,             # 2*pi salvage: n in [-range, +range]
    # conservative mode: salvaged records contribute only within the
    # wavelength range covered by trusted (F/R-agreeing) spacings, and
    # unverified single-direction records beyond that range are dropped.
    # Deep Vs then relies exclusively on cross-validated data (smaller
    # resolution depth, lower-bound estimate of deep stiffness).
    "conservative_deep": False,
    # composite
    "n_bins": 28,
    "sigma_floor": 0.08,           # relative
    "trust_avg": 1.0,
    "trust_validated": 1.5,
    "trust_unverified": 2.0,
    # forward model
    "poisson": 0.35,
    "rho": 1.85,                   # t/m3
    # inversion
    "fine_h": 0.4,                 # m, layer thickness
    "vs_min": 50.0,
    "vs_max": 1200.0,
    "rough_weight": 40.0,          # on mean squared d(ln Vs)
    "resolution_divisor": 2.5,     # z_res = lambda_max / this
    # joint inversion
    "target_r2": 0.95,
    "wp_sweep": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "refusal_sigma_factor": 1.3,
    "prior_sigma_min": 0.15,
}
