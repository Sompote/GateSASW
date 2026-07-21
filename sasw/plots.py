"""Result figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .forward import forward_vph
from .inversion import profile_at, resolution_depth


def profile_steps(thick, vs, zmax):
    z = np.concatenate([[0], np.repeat(np.cumsum(thick), 2), [zmax]])
    return np.repeat(vs, 2), z


def plot_results(out_dir, site, wl, vph, sp, wlc, vc, sig, thick, vs, rms,
                 P, spt=None, best_eq=None, thick_c=None, vs_c=None,
                 rms_c=None, r2=None):
    """One summary figure: dispersion data + composite, model fit, profile
    (+ optional joint inversion and SPT reference)."""
    zres = resolution_depth(wlc, P)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    ax = axes[0]
    for d in np.unique(sp):
        m = sp == d
        ax.scatter(wl[m], vph[m], s=8, alpha=0.4, label=f"d={d:g} m")
    ax.errorbar(wlc, vc, yerr=sig, fmt="k-o", ms=4, lw=1.8, capsize=2,
                label="composite $\\pm\\sigma$")
    ax.set(xscale="log", xlabel="Wavelength (m)", ylabel="$V_{ph}$ (m/s)",
           title=f"{site} — dispersion data")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    f_obs = vc / wlc
    ax.plot(wlc, vc, "ko-", ms=4, label="observed composite")
    ax.plot(wlc, forward_vph(thick, vs, f_obs, P), "b--s", ms=4,
            label=f"model (RMS {rms*100:.1f}%)")
    if vs_c is not None:
        ax.plot(wlc, forward_vph(thick_c, vs_c, f_obs, P), "r:", lw=2,
                label=f"joint model (RMS {rms_c*100:.1f}%)")
    ax.set(xscale="log", xlabel="Wavelength (m)", ylabel="$V_{ph}$ (m/s)",
           title=f"{site} — dispersion fit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    zmax = (np.sum(thick_c) if thick_c is not None else np.sum(thick)) + 1.5
    v, z = profile_steps(thick, vs, zmax)
    ax.plot(v, z, "-", color="navy", lw=2.2, label="$V_s$ (dispersion only)")
    if vs_c is not None:
        v2, z2 = profile_steps(thick_c, vs_c, zmax)
        lbl = "$V_s$ (joint inversion" + \
            (f", R$^2$={r2:.2f})" if r2 is not None else ")")
        ax.plot(v2, z2, "-", color="crimson", lw=2.2, label=lbl)
    if zres < zmax:
        ax.axhspan(zres, zmax, color="grey", alpha=0.15, zorder=0)
        ax.axhline(zres, color="grey", ls=":", lw=1)
        ax.text(0.98, zres + 0.1, "resolution depth", fontsize=8,
                color="dimgray", ha="right", va="top",
                transform=ax.get_yaxis_transform())
    if spt and best_eq:
        name, a, b, s_ln = best_eq
        for d0, d1, N, status in spt:
            mid = (d0 + d1) / 2
            p = a * N ** b
            col = "teal" if status != "refusal" else "coral"
            ax.plot([p * np.exp(-s_ln), p * np.exp(s_ln)], [mid, mid],
                    color=col, lw=5, alpha=0.35, solid_capstyle="round")
            ax.plot(p, mid, "o", mfc="white", mec=col, mew=2, ms=8)
            ax.annotate(f"N={N:g}", (p * np.exp(s_ln), mid),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=col, va="center")
        ax.plot([], [], "o-", color="teal", mfc="white",
                label=f"{name}: $V_s={a:g}N^{{{b:g}}}$")
    ax.set(xlabel="$V_s$ (m/s)", ylabel="Depth (m)", ylim=(zmax, 0))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle(f"SASW analysis — {site}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / f"{site}_sasw_results.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
