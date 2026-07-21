# GateSASW — Quality-Gated SASW Processing & Inversion

A deterministic, borehole-independent pipeline that turns raw two-receiver
Spectral-Analysis-of-Surface-Waves (SASW) field records into a shear-wave
velocity (Vs) profile — with quantitative quality gates, a self-reported
resolution depth, and optional verification against Standard Penetration
Test (SPT) data.

Every processing decision is a numerical rule, so the same pipeline runs
unattended over one site or a hundred, produces the identical result for the
same input, and never lets the borehole influence the surface-wave profile.

```
raw Hxy / coherence spectra  (two receivers, F and R shots, all spacings)
  │
  ├─ Stage A  phase unwrap + sign convention + Savitzky–Golay smoothing
  ├─ Stage B  masking (coherence, λ∈[d/2, 3d]) + 4 quality gates:
  │              1  phase monotonicity   ρs(f,φ) ≥ 0.5
  │              2  forward/reverse agreement ≤ 20%  → phase averaging
  │              3  integer-cycle (2π) recovery, cross-validated
  │              4  continuity screen (single-direction records)
  │           + conservative validated-range restriction
  ├─ Stage C  trust-weighted composite dispersion curve
  ├─ Stage D  regularized inversion (exact Thomson–Haskell via disba)
  │              → Vs profile + resolution depth  z_res ≈ λmax/2.5
  ├─ Stage E  [optional] independent SPT verification / correlation selection
  └─ Stage F  [optional] SPT-constrained joint inversion
```

---

## Why this exists

Two-receiver SASW is a phase-difference measurement: the whole Vs profile
comes from the frequency-dependent phase lag between two geophones. That
makes it powerful but fragile — receiver-coupling errors, phase-unwrapping
cycle jumps, and body-wave / higher-mode contamination can corrupt a record
in ways that a coherence check alone does not catch. Conventional processing
leaves the accept/reject decision to an expert analyst, which makes results
hard to reproduce and easy to bias.

GateSASW replaces that judgment with explicit rules, and — crucially — knows
where to stop: it computes a per-site **resolution depth** below which the
profile is a regularized extrapolation rather than a measurement, and it
never compares that extrapolated part against borehole data.

---

## Install

Requires Python ≥ 3.9.

```bash
git clone https://github.com/Sompote/GateSASW.git
cd GateSASW
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, and
[`disba`](https://github.com/keurfonluu/disba) (Numba-accelerated
Thomson–Haskell surface-wave solver).

---

## Input data layout

One folder per field record inside a site directory. Spacing and shot
direction are parsed from the folder name using the pattern
`H<spacing>m<F|R>...` (e.g. `H4mR_HM.trf` = 4 m spacing, reverse shot):

```
MY_SITE/
├── H1mF_HM.trf/Run00001/ASCII/Hxysv00000.txt   # transfer function: freq, Re, Im
│                             └ Cxysv00000.txt   # coherence:         freq, γ²
├── H1mR_HM.trf/Run*/ASCII/...
├── H2mF_HM.trf/...
├── H4mF_HM.trf/...   H4mR_HM.trf/...
├── H8mF_DW.trf/...   H8mR_HM.trf/...
├── H16mF_DW.trf/...
└── H20mF_DW.trf/...
```

- Any `Run*` folder number works; the first (sorted) is used.
- Multiple records for the same spacing+direction (e.g. hammer *and*
  drop-weight) are allowed — the highest-quality one is selected
  automatically.
- The ASCII files are the standard dynamic-signal-analyzer export: a header
  block ending in a line starting with `Hz`, then whitespace-separated
  numeric columns. The reader tolerates ragged rows and locks the column
  count to the first valid data line.

### SPT file (optional)

CSV with one row per SPT interval; `status` is `actual` (measured N) or
`refusal` (N_eq > 100, extrapolated blow count):

```csv
depth_top_m,depth_bot_m,N,status
1.50,1.95,20,actual
3.00,3.45,60,actual
4.50,4.88,130,refusal
```

Example SPT files for two boreholes are in `examples/`.

---

## Usage

### Single site

```bash
# dispersion curve + Vs profile (surface waves only)
python run_sasw.py /path/to/MY_SITE

# conservative deep mode: deep Vs from cross-validated data only (recommended)
python run_sasw.py /path/to/MY_SITE --conservative

# add SPT: correlation selection + SPT-constrained joint inversion
python run_sasw.py /path/to/MY_SITE --conservative --spt examples/BHPB-68_spt.csv

# custom output folder and parameters
python run_sasw.py /path/to/MY_SITE --out results \
    --poisson 0.35 --rho 1.85 --fine-h 0.4 --target-r2 0.95

# override any default via JSON
python run_sasw.py /path/to/MY_SITE --params my_params.json
```

### Many sites (batch)

Recursively finds every folder containing `*.trf` records, de-duplicates by
site name (canonical folders preferred over variant copies), and processes
each:

```bash
python batch_run.py /path/to/PROJECT_ROOT --out batch_results --conservative
```

Produces `batch_results/summary.csv` (one row per site: λmax, z_res, RMS,
Vs range, Vs30-style average), `batch_results/all_profiles.png` (overlay of
every profile with resolution depths marked), and a per-site subfolder.

### As a library

```python
from sasw import (DEFAULTS, auto_detect_tests, collect_dispersion,
                  build_composite, invert, resolution_depth)

P = dict(DEFAULTS); P["conservative_deep"] = True
data = "/path/to/MY_SITE"
tests = auto_detect_tests(data)
wl, vph, sp, tr, report = collect_dispersion(data, tests, P)
wlc, vc, sig = build_composite(wl, vph, tr, P)
thick, vs, rms = invert(wlc, vc, sig, P)
print("resolution depth:", resolution_depth(wlc, P), "m")
```

---

## Outputs (default `<site>/sasw_out/`)

| File | Content |
|---|---|
| `dispersion_composite.csv` | composite dispersion curve: wavelength, Vph, σ |
| `vs_profile.csv` | Vs profile from dispersion data only (layer top/bottom/Vs) |
| `vs_profile_joint.csv` | Vs profile from SPT-constrained joint inversion (with `--spt`) |
| `<site>_sasw_results.png` | 3-panel figure: dispersion data, model fit, Vs profile |
| `report.txt` | full QC log (every accept/repair/reject with its statistic) + summary |

The `report.txt` is the audit trail: for every record it states whether it
was averaged, salvaged (with the 2π offsets found), kept single-direction,
or rejected (with the failing statistic).

---

## The resolution depth — read this before using deep values

The profile is only a *measurement* down to

```
z_res ≈ λ_max / 2.5
```

where λ_max is the longest reliable wavelength that survived quality control.
Below z_res the inversion still returns numbers, but they are controlled by
the smoothness regularization, not by the data — the outputs mark this zone
in grey and the batch summary reports `z_res` per site. To image deeper you
need longer reliable wavelengths, i.e. **forward *and* reverse shots at the
long spacings (16–20 m) with a strong low-frequency source**, so gates 2–3
can validate them.

---

## Method summary

| Stage | What it does | Key rule |
|---|---|---|
| A. Phase | unwrap cross-power phase, fix sign, smooth | Savitzky–Golay (11, 3) |
| B. Mask | keep good band | γ² ≥ 0.95 (d ≤ 4 m) / 0.90 (d ≥ 8 m); λ ∈ [d/2, 3d] |
| B. Gate 1 | reject body-wave records | Spearman ρ(f, φ) ≥ 0.5 + local-slope filter |
| B. Gate 2 | forward/reverse reciprocity | Δ_FR ≤ 20% → average phases (trust ×1.0) |
| B. Gate 3 | repair unwrapping errors | search φ + 2πn; accept iff F/R restored **and** cross-validated ≤ 25% (trust ×1.5) |
| B. Gate 4 | drop mode-contaminated records | > 50% step at spectral join → reject (else trust ×2.0) |
| B. Conservative | keep deep data honest | non-trusted records restricted to trusted-spacing wavelength range |
| C. Composite | merge spacings | log-λ median bins, IQR rejection, σ = max(std, 8%)·trust |
| D. Forward | Rayleigh dispersion | exact Thomson–Haskell / Dunkin fast-delta (disba) |
| D. Inversion | fit Vs profile | fixed 0.4 m layers, log-Vs, Nelder–Mead + log-roughness |
| D. z_res | state validity limit | λ_max / 2.5, perturbation-verifiable |
| E. Verify | independent SPT check | soils two-sided (95% band); refusal one-sided lower bound; only z ≤ z_res |
| F. Joint | borehole-consistent model | J = χ²_disp + w_p·χ²_prior + w_r·roughness, w_p tuned to R² target |

`--spt` enables Stages E–F. Without it the pipeline stops at the
borehole-independent Vs profile. **Note:** the joint inversion (Stage F) uses
the SPT correlation as a prior below z_res, so its deep values are
SPT-informed by construction — use it as an engineering deliverable, not as
independent validation of SASW against SPT.

---

## Parameters (`sasw/config.py`)

All defaults live in `DEFAULTS`; override via `--params my_params.json` or
the named CLI flags. The most relevant:

| Key | Default | Meaning |
|---|---|---|
| `coh_threshold_short` / `coh_threshold_long` | 0.95 / 0.90 | coherence threshold for d ≤ 4 m / d ≥ 8 m |
| `lambda_min_ratio` / `lambda_max_ratio` | 0.5 / 3.0 | wavelength window λ ∈ [d·min, d·max] |
| `min_pts` | 10 | minimum masked points for a record to be considered |
| `spearman_min` | 0.5 | Gate 1 phase-monotonicity threshold |
| `fr_agree` | 0.20 | Gate 2 forward/reverse agreement threshold |
| `cross_dev` | 0.25 | Gate 3 cross-spacing validation threshold |
| `offset_range` | 2 | 2π salvage searches n ∈ [−2, +2] |
| `continuity_step` | 0.50 | Gate 4 maximum step at the spectral join |
| `conservative_deep` | false | restrict non-trusted records to validated range (`--conservative`) |
| `poisson` / `rho` | 0.35 / 1.85 | Poisson ratio / density (t/m³) for the forward model |
| `fine_h` | 0.4 | inversion layer thickness (m) |
| `vs_min` / `vs_max` | 50 / 1200 | velocity bounds (m/s) |
| `resolution_divisor` | 2.5 | z_res = λ_max / this |
| `target_r2` | 0.95 | joint-inversion R² target |

---

## Repository map

```
run_sasw.py          CLI for one site
batch_run.py         CLI for many sites + summary
sasw/
  config.py          DEFAULTS parameter dictionary
  io.py              record auto-detection + robust ASCII reader
  phase.py           phase processing + dispersion extraction (Stage A/B)
  qc.py              quality gates, 2π salvage, conservative restriction
  composite.py       trust-weighted composite curve (Stage C)
  forward.py         Thomson–Haskell forward model via disba (Stage D)
  inversion.py       regularized inversion, resolution depth, joint inversion
  correlations.py    19 Vs–N equations + site-appropriate selection
  plots.py           result figures
examples/            example SPT CSVs
batch_results/       worked outputs for 13 bridge sites
```

---

## Troubleshooting

- **"all records rejected by QC"** — the site's records failed the gates.
  Check `report.txt` for the reasons; common causes are low coherence or
  non-monotone phase (body-wave contamination). Loosen `spearman_min` or
  `coh_threshold_*` only if you understand the trade-off.
- **Very shallow z_res (~2 m)** — the long-spacing records failed QC. This is
  honest behaviour, not a bug; re-acquire d ≥ 16 m in both shot directions.
- **High composite RMS (> 20%)** — inspect `<site>_sasw_results.png`; the
  dispersion cloud is scattered. The profile is still returned but its fine
  detail should be treated cautiously.
- **`disba` import error** — `pip install disba`; it needs a working Numba /
  LLVM stack.

---

## Citation

If you use this software, please cite the accompanying paper (SASW
quality-gated framework, KMUTT / University of Phayao) and the disba solver
(Luu, 2021). Key method references:

Nazarian & Stokoe (1984); Heisey et al. (1982); Al-Hunaidi (1993);
Rosenblad & Bertel (2008); Joh et al. (2023); Thomson (1950); Haskell (1953);
Dunkin (1965); Buchen & Ben-Hador (1996); Foti et al. (2014, 2018);
Wair et al. (2012); Imai & Tonouchi (1982) and the other Vs–N correlations
listed in `sasw/correlations.py`.

## License

MIT — see [LICENSE](LICENSE).
