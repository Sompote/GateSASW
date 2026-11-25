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
    # --- revisions (Nov 2026; defaults reproduce the original pipeline) ---
    "run_select": "first",         # first | last | all  (all = V1 run recovery)
    "qc_mode": "legacy",           # legacy | consensus (V3)
    "adaptive_bins": False,        # composite bin count follows point count
    "inv_mode": "legacy",          # legacy | graded (V2: graded layers, L-BFGS-B)
    "h_growth": 1.15,              # graded layers: thickness growth per layer
    "h_max": 2.0,                  # graded layers: maximum thickness (m)
    "model_div": 2.0,              # graded model depth = lambda_max / model_div
    "unwrap_mode": "legacy",       # legacy | chain (V5 cross-spacing unwrapping)
    "chain_anchor_lo": 0.25,       # anchor phase range [-lo, 1-lo) cycle
    "chain_ambiguous": 0.35,       # flag a segment above this residual (cycle)
    "chain_strict": False,         # reject flagged records instead of keeping
    "top_divisor": 3.0,
    "irf": False,                  # impulse-response gating of Hxy before unwrapping
    "irf_vmin": 60.0,              # slowest admissible wave: gate ends at d / irf_vmin
    "irf_taper_s": 0.010, "irf_taper_frac": 0.25, "irf_lead_s": 0.004,
    "coh_share_min": 0.30,         # site flagged low-coherence below this share            # z_min = lambda_min / this
    "forward": "fundamental",      # fundamental | effective (V4 mode-corrected)
    "eff_modes": 4, "eff_iters": 3, "eff_rmin": 0.7, "eff_rmax": 1.6,
    # joint inversion
    "target_r2": 0.95,
    "wp_sweep": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "refusal_sigma_factor": 1.3,
    "prior_sigma_min": 0.15,
}
