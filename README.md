# sasw — Quality-Controlled SASW Processing & Inversion

Complete workflow from raw two-receiver SASW spectra to the experimental
dispersion curve and shear-wave velocity (Vs) profile, with optional
SPT-constrained joint inversion.

```
raw Hxy/coherence spectra
   └─ phase unwrapping + smoothing
       └─ QC gates (coherence, phase monotonicity, F/R agreement,
          2π-offset salvage, cross-spacing validation, continuity)
           └─ trust-weighted composite dispersion curve
               └─ regularized inversion (exact Thomson–Haskell forward, disba)
                   └─ Vs profile + resolution depth
                       └─ [optional] Vs–N correlation selection
                           └─ SPT-constrained joint inversion (R² target)
```

## Install

```bash
pip install -r requirements.txt
```

## Data layout

One folder per record inside your site data directory. Spacing and shot
direction are parsed from the folder name (`H<spacing>m<F|R>...`):

```
MY_SITE/
├── H1mF_HM.trf/Run00001/ASCII/Hxysv00000.txt   # transfer function (Re, Im)
│                             └ Cxysv00000.txt  # coherence
├── H1mR_HM.trf/Run*/ASCII/...
├── H2mF_HM.trf/...
├── H4mR_HM.trf/...
└── H20mF_DW.trf/...
```

Any `Run*` number works; the first is used.

## Usage

```bash
# dispersion curve + Vs profile
python run_sasw.py /path/to/MY_SITE

# with SPT borehole data: correlation selection + joint inversion
python run_sasw.py /path/to/MY_SITE --spt my_spt.csv

# custom output folder / parameters
python run_sasw.py /path/to/MY_SITE --out results --target-r2 0.95 \
    --poisson 0.35 --rho 1.85 --fine-h 0.4
python run_sasw.py /path/to/MY_SITE --params my_params.json
```

SPT CSV format (`status` is `actual` or `refusal`):

```csv
depth_top_m,depth_bot_m,N,status
1.50,1.95,20,actual
3.00,3.45,60,actual
4.50,4.88,130,refusal
```

## Outputs (in `<site>/sasw_out/` by default)

| File | Content |
|---|---|
| `dispersion_composite.csv` | composite dispersion curve with uncertainty |
| `vs_profile.csv` | Vs profile (dispersion data only) |
| `vs_profile_joint.csv` | Vs profile from SPT-constrained joint inversion |
| `<site>_sasw_results.png` | dispersion data, model fit, Vs profiles |
| `report.txt` | full QC log and summary |

## Method

- **Phase processing:** cross-power phase unwrapping with sign-convention
  fixing and Savitzky–Golay smoothing; V_ph = 2πfd/φ, λ = 2πd/φ.
- **Masking:** coherence γ² ≥ 0.95 (d ≤ 4 m) / 0.90 (d ≥ 8 m); λ ∈ [d/2, 3d];
  positive local phase slope.
- **QC gates:**
  1. record gate — ≥ 10 masked points and Spearman ρ(f, φ) ≥ 0.5 (rejects
     phase plateaus/reversals: body-wave contamination that passes the
     coherence screen);
  2. forward/reverse agreement ≤ 20% → phase averaging;
  3. disagreeing pairs → integer-cycle (2π) offset salvage, accepted only if
     agreement is restored **and** the pair cross-validates (≤ 25%) against a
     trusted spacing;
  4. single-direction records carry doubled uncertainty and must pass a 50%
     continuity gate at the spectral join.
- **Composite:** log-wavelength median binning, IQR outlier rejection,
  σ = max(scatter, 8%)·trust.
- **Forward model:** fundamental-mode Rayleigh dispersion via the
  Thomson–Haskell / Dunkin fast-delta scheme (open-source
  [disba](https://github.com/keurfonluu/disba)).
- **Inversion:** fixed thin layers (0.4 m default), velocities-only in
  log-space, Nelder–Mead with log-roughness regularization.
- **Resolution depth:** z_res ≈ λ_max/2.5; the profile below is
  regularization-controlled, not data-constrained.
- **Correlation selection:** 19 published Vs = a·N^b equations ranked by log
  residual against the *resolved-zone* SASW profile (non-circular); the best
  becomes the joint-inversion prior with its calibration residual as prior
  uncertainty (×1.3 for refusal intervals).
- **Joint inversion:** J = χ²_dispersion + w_p·χ²_prior + w_r·roughness;
  w_p increased stepwise until the SASW-vs-prediction R² reaches the target
  (default 0.95). The dispersion RMS is reported at every step so the
  trade-off is explicit.

## References

Nazarian & Stokoe (1984); Heisey et al. (1982); Rosenblad & Bertel (2008);
Joh et al. (2023); Thomson (1950); Haskell (1953); Dunkin (1965);
Buchen & Ben-Hador (1996); Luu — disba (2021); Foti et al. (2014, 2018);
Iyisan (1996) and the other 18 Vs–N correlations listed in
`sasw/correlations.py`.

## License

MIT
