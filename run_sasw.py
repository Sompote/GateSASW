#!/usr/bin/env python3
"""SASW analysis pipeline: raw two-receiver spectra -> dispersion curve ->
Vs profile (+ optional SPT-constrained joint inversion).

Usage:
    python run_sasw.py DATA_DIR [options]

DATA_DIR contains one folder per record, e.g. H4mR_HM.trf/Run*/ASCII/ with
Hxysv00000.txt (transfer function) and Cxysv00000.txt (coherence). Spacing
and direction are parsed from the folder name (H<spacing>m<F|R>...).

Examples:
    python run_sasw.py /path/to/RB249-P1
    python run_sasw.py /path/to/RB249-P1 --spt examples/BHPB-68_spt.csv
    python run_sasw.py /path/to/site --out results --target-r2 0.95
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path

from sasw import (DEFAULTS, auto_detect_tests, read_spt_csv,
                  collect_dispersion, build_composite, invert,
                  joint_inversion_sweep, profile_at, resolution_depth,
                  select_best, build_priors)
from sasw.plots import plot_results


def export_profile(path, thick, vs):
    zi = np.concatenate([[0], np.cumsum(thick)])
    with open(path, "w") as f:
        f.write("Layer,Depth_top_m,Depth_bot_m,Vs_m_per_s\n")
        for k in range(len(vs)):
            bot = f"{zi[k + 1]:.2f}" if k < len(thick) else "inf"
            f.write(f"{k + 1},{zi[k]:.2f},{bot},{vs[k]:.1f}\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="folder containing the .trf record folders")
    ap.add_argument("--out", default=None,
                    help="output folder (default: DATA_DIR/sasw_out)")
    ap.add_argument("--spt", default=None,
                    help="SPT CSV (depth_top_m,depth_bot_m,N,status) enabling "
                         "correlation selection + joint inversion")
    ap.add_argument("--site", default=None, help="site label for outputs")
    ap.add_argument("--target-r2", type=float, default=None)
    ap.add_argument("--poisson", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--fine-h", type=float, default=None,
                    help="inversion layer thickness (m)")
    ap.add_argument("--conservative", action="store_true",
                    help="deep Vs from cross-validated data only: salvaged/"
                         "unverified records restricted to the wavelength "
                         "range covered by trusted spacings")
    ap.add_argument("--params", default=None,
                    help="JSON file overriding any default parameter")
    args = ap.parse_args()

    P = dict(DEFAULTS)
    if args.params:
        P.update(json.loads(Path(args.params).read_text()))
    for key, val in [("target_r2", args.target_r2),
                     ("poisson", args.poisson), ("rho", args.rho),
                     ("fine_h", args.fine_h)]:
        if val is not None:
            P[key] = val
    if args.conservative:
        P["conservative_deep"] = True

    data_dir = Path(args.data_dir)
    site = args.site or data_dir.name
    out = Path(args.out) if args.out else data_dir / "sasw_out"
    out.mkdir(parents=True, exist_ok=True)

    print(f"SASW analysis — {site}")
    print("=" * 60)

    # 1. detect records
    tests = auto_detect_tests(data_dir, P.get("run_select", "first"))
    if not tests:
        sys.exit(f"no record folders (H<d>m<F|R>*.trf/Run*/ASCII) found in "
                 f"{data_dir}")
    print(f"  records detected: {len(tests)} "
          f"(spacings {sorted(set(t[1] for t in tests))} m)")

    # 2. dispersion extraction with QC
    print("  quality control:")
    wl, vph, sp, tr, report = collect_dispersion(data_dir, tests, P)

    from sasw import coherent_share
    cshare, clow = coherent_share(data_dir, tests, P)
    print(f"  coherent share of the records: {cshare:.2f}"
          + ("  ** LOW-COHERENCE SITE: profile is of reduced reliability **"
             if clow else ""))
    report.append(f"coherent share {cshare:.2f}"
                  + (" (LOW-COHERENCE SITE)" if clow else ""))

    # 3. composite curve
    wlc, vc, sig = build_composite(wl, vph, tr, P)
    zres = resolution_depth(wlc, P)
    print(f"  composite: {len(wlc)} pts, lambda {wlc.min():.1f}-"
          f"{wlc.max():.1f} m  ->  resolution depth ~{zres:.1f} m")
    with open(out / "dispersion_composite.csv", "w") as f:
        f.write("Wavelength_m,Phase_Velocity_m_per_s,Sigma_m_per_s\n")
        for a, b, c in zip(wlc, vc, sig):
            f.write(f"{a:.3f},{b:.1f},{c:.1f}\n")

    # 4. unconstrained inversion
    print("  inversion (dispersion data only):")
    if P.get("forward") == "effective":
        from sasw.effective import bin_spacing
        from sasw.inversion import invert_effective
        thick, vs, rms, _, _ = invert_effective(wlc, vc, sig,
                                                bin_spacing(wl, sp, wlc), P)
    else:
        thick, vs, rms = invert(wlc, vc, sig, P)
    print(f"    fit RMS = {rms*100:.1f}%, {len(vs)} layers (top {P['fine_h']} m)")
    export_profile(out / "vs_profile.csv", thick, vs)

    # 5. optional SPT: correlation selection + joint inversion
    spt = best_eq = None
    thick_c = vs_c = rms_c = r2 = None
    if args.spt:
        spt = read_spt_csv(args.spt)
        cal = [(N, profile_at(thick, vs, (d0 + d1) / 2))
               for d0, d1, N, st in spt if (d0 + d1) / 2 <= zres]
        if len(cal) < 2:
            print("  WARNING: fewer than 2 SPT depths inside the resolved "
                  "zone — correlation selection skipped, joint inversion "
                  "uses the pooled default (Fatehnia et al. 2015)")
            best_eq = ("Fatehnia et al. (2015)", 77.1, 0.355, 0.30)
        else:
            print(f"  correlation selection ({len(cal)} resolved-zone "
                  f"calibration points):")
            name, a, b, s_ln, _ = select_best(cal)
            best_eq = (name, a, b, s_ln)
            print(f"    selected: {name}  Vs = {a:g} N^{b:g} "
                  f"(sigma_ln {s_ln:.2f})")
        priors = build_priors(spt, best_eq, P)
        reference = [((d0 + d1) / 2, best_eq[1] * N ** best_eq[2])
                     for d0, d1, N, st in spt]
        print("  joint inversion (prior weight sweep):")
        thick_c, vs_c, rms_c, w_p, r2, hist = joint_inversion_sweep(
            wlc, vc, sig, P, priors, reference)
        print(f"    final: w_p={w_p:g}, R2={r2:.3f}, "
              f"dispersion RMS={rms_c*100:.1f}%")
        export_profile(out / "vs_profile_joint.csv", thick_c, vs_c)

    # 6. figure + report
    fig_path = plot_results(out, site, wl, vph, sp, wlc, vc, sig, thick, vs,
                            rms, P, spt=spt, best_eq=best_eq,
                            thick_c=thick_c, vs_c=vs_c, rms_c=rms_c, r2=r2)
    with open(out / "report.txt", "w") as f:
        f.write(f"SASW analysis report — {site}\n{'=' * 60}\n\nQC log:\n")
        f.write("\n".join(report))
        f.write(f"\n\ncomposite: {len(wlc)} pts, lambda "
                f"{wlc.min():.1f}-{wlc.max():.1f} m\n"
                f"resolution depth ~{zres:.1f} m\n"
                f"unconstrained fit RMS {rms*100:.1f}%\n")
        if best_eq:
            f.write(f"selected correlation: {best_eq[0]} "
                    f"(Vs = {best_eq[1]:g} N^{best_eq[2]:g})\n")
        if r2 is not None:
            f.write(f"joint inversion: R2={r2:.3f}, "
                    f"RMS={rms_c*100:.1f}%\n")
    print(f"  outputs -> {out}")
    print(f"    {fig_path.name}, dispersion_composite.csv, vs_profile.csv"
          + (", vs_profile_joint.csv" if vs_c is not None else "")
          + ", report.txt")


if __name__ == "__main__":
    main()
