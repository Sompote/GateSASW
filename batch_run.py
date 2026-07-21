#!/usr/bin/env python3
"""Batch-run the SASW pipeline over many sites and collect a summary.

Usage:
    python batch_run.py ROOT_DIR --out OUTDIR [--conservative]

Recursively finds every folder that directly contains *.trf record
sub-folders, de-duplicates by site name (folder basename), runs the
dispersion-to-Vs pipeline on each, and writes:
    OUTDIR/<site>/          per-site outputs (profile, composite, figure)
    OUTDIR/summary.csv      one row per site
    OUTDIR/all_profiles.png depth-Vs overlay of every site
"""
import argparse
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sasw import (DEFAULTS, auto_detect_tests, collect_dispersion,
                  build_composite, invert, resolution_depth, profile_at)
from sasw.plots import plot_results, profile_steps


def find_sites(root):
    """Return {site_name: data_dir} for every folder containing *.trf
    records, de-duplicated by basename. Canonical folders are preferred
    over hyperparameter-variant copies."""
    candidates = {}
    for trf in sorted(Path(root).rglob("*.trf")):
        d = trf.parent
        if not list(d.glob("*.trf")):
            continue
        candidates.setdefault(d.name, []).append(d)

    def rank(path):
        s = str(path).lower()
        return (("hyperparameter" in s or "varied" in s),  # variants last
                len(str(path)))                             # shorter first
    return {name: sorted(paths, key=rank)[0]
            for name, paths in candidates.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="batch_out")
    ap.add_argument("--conservative", action="store_true")
    args = ap.parse_args()

    P = dict(DEFAULTS)
    if args.conservative:
        P["conservative_deep"] = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = find_sites(args.root)
    print(f"Found {len(sites)} unique sites\n" + "=" * 60)

    rows = []
    profiles = {}
    for name, data_dir in sites.items():
        print(f"\n### {name}")
        site_out = out / name
        site_out.mkdir(exist_ok=True)
        try:
            tests = auto_detect_tests(data_dir)
            wl, vph, sp, tr, report = collect_dispersion(
                data_dir, tests, P, log=lambda m: None)
            wlc, vc, sig = build_composite(wl, vph, tr, P)
            zres = resolution_depth(wlc, P)
            thick, vs, rms = invert(wlc, vc, sig, P)
            profiles[name] = (thick, vs, zres)

            zi = np.concatenate([[0], np.cumsum(thick)])
            with open(site_out / "vs_profile.csv", "w") as f:
                f.write("Layer,Depth_top_m,Depth_bot_m,Vs_m_per_s\n")
                for k in range(len(vs)):
                    b = f"{zi[k+1]:.2f}" if k < len(thick) else "inf"
                    f.write(f"{k+1},{zi[k]:.2f},{b},{vs[k]:.1f}\n")
            with open(site_out / "dispersion_composite.csv", "w") as f:
                f.write("Wavelength_m,Phase_Velocity_m_per_s,Sigma_m_per_s\n")
                for a, b, c in zip(wlc, vc, sig):
                    f.write(f"{a:.3f},{b:.1f},{c:.1f}\n")
            plot_results(site_out, name, wl, vph, sp, wlc, vc, sig,
                         thick, vs, rms, P)

            vs30 = None
            if zres >= 5:
                # harmonic (travel-time) average over the finite layers
                # down to min(zres, 30 m); vs has one more entry (half-space)
                # than thick, so align to the layer velocities vs[:len(thick)]
                zc = np.cumsum(thick)
                dz = thick
                vlayer = vs[:len(thick)]
                zmax = min(30.0, zres, float(zc[-1]))
                sel = zc <= zmax
                if sel.any():
                    vs30 = np.sum(dz[sel]) / np.sum(dz[sel] / vlayer[sel])
            rows.append(dict(
                site=name, n_records=len(tests),
                lambda_max=round(wlc.max(), 1),
                z_res=round(zres, 1), rms_pct=round(rms * 100, 1),
                vs_surface=round(vs[0]),
                vs_at_zres=round(profile_at(thick, vs, zres)),
                vs_halfspace=round(vs[-1]),
                vsz_avg=round(vs30) if vs30 else ""))
            print(f"    OK  z_res={zres:.1f} m  RMS={rms*100:.1f}%  "
                  f"Vs {vs[0]:.0f}->{vs[-1]:.0f} m/s")
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            rows.append(dict(site=name, n_records=0, lambda_max="",
                             z_res="", rms_pct="", vs_surface="",
                             vs_at_zres="", vs_halfspace="", vsz_avg="",
                             error=str(e)))

    # summary CSV
    cols = ["site", "n_records", "lambda_max", "z_res", "rms_pct",
            "vs_surface", "vs_at_zres", "vs_halfspace", "vsz_avg"]
    with open(out / "summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # overlay figure
    ok = [(n, p) for n, p in profiles.items()]
    if ok:
        ncol = 4
        nrow = int(np.ceil(len(ok) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow),
                                 sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (name, (thick, vs, zres)) in zip(axes, sorted(ok)):
            zmax = min(float(np.sum(thick)), 12)
            v, z = profile_steps(thick, vs, zmax)
            ax.plot(v, z[:len(v)], "-", color="navy", lw=1.8, zorder=3)
            if zres < zmax:
                ax.axhspan(zres, zmax, color="grey", alpha=0.15, zorder=0)
                ax.axhline(zres, color="grey", ls=":", lw=0.8)
            ax.set(title=f"{name}  (z$_{{res}}$={zres:.1f} m)",
                   xlim=(0, 1000), ylim=(zmax, 0))
            ax.grid(alpha=0.3)
        for ax in axes[len(ok):]:
            ax.axis("off")
        for ax in axes[::ncol]:
            ax.set_ylabel("Depth (m)")
        for ax in axes[-ncol:]:
            ax.set_xlabel("Vs (m/s)")
        fig.suptitle("SASW Vs profiles — all bridge sites "
                     "(grey = below resolution depth)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "all_profiles.png", dpi=150, bbox_inches="tight")

    print("\n" + "=" * 60)
    print(f"Summary -> {out/'summary.csv'}")
    print(f"Overlay -> {out/'all_profiles.png'}")
    ok_n = sum(1 for r in rows if r.get("z_res") != "")
    print(f"{ok_n}/{len(rows)} sites processed successfully")


if __name__ == "__main__":
    main()
