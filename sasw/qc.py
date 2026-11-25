"""Quality-control gates and quality-hierarchy dispersion assembly.

Acceptance hierarchy (Nazarian-Stokoe criterion enforced, plus salvage):
1. per-record gate: >= min_pts masked points AND monotone phase trend
2. F and R both valid and agreeing        -> phase average   (trust 1.0)
3. disagreeing F/R pair: try integer-cycle (2*pi) offset salvage, accepted
   only if F/R agreement is restored AND the corrected pair cross-validates
   against a trusted overlapping spacing  -> salvaged        (trust 1.5)
4. otherwise keep the single direction validated by a trusted spacing;
   if none validates, REJECT the spacing
5. single-direction records: keep unless contradicted by a trusted overlap
   (trust 2.0); reject if their curve steps > continuity_step off the
   neighboring accepted curve (body-wave / higher-mode contamination)
"""
import numpy as np
from itertools import product
from scipy.stats import spearmanr

from .io import read_record
from .phase import (process_phase, process_phase_segmented,
                    phase_to_dispersion)


def record_quality(wl, vph, f, P):
    if len(wl) < P["min_pts"]:
        return len(wl), np.nan
    rho = spearmanr(f, f / vph).statistic   # ~ phase vs frequency
    return len(wl), rho


def fr_disagreement(wlF, vF, wlR, vR):
    lo, hi = max(wlF.min(), wlR.min()), min(wlF.max(), wlR.max())
    if lo >= hi:
        return np.inf
    wt = np.linspace(lo, hi, 20)
    a = np.interp(wt, np.sort(wlF), vF[np.argsort(wlF)])
    b = np.interp(wt, np.sort(wlR), vR[np.argsort(wlR)])
    return np.mean(np.abs(a - b) / ((a + b) / 2))


def overlap_dev(wlA, vA, wlB, vB):
    lo, hi = max(wlA.min(), wlB.min()), min(wlA.max(), wlB.max())
    if lo >= hi:
        return np.nan
    wt = np.linspace(lo, hi, 20)
    a = np.interp(wt, np.sort(wlA), vA[np.argsort(wlA)])
    b = np.interp(wt, np.sort(wlB), vB[np.argsort(wlB)])
    return np.median(np.abs(a - b) / b)


def average_pair(fD, rD, spacing, P):
    """Average F and R phases on a common frequency grid, then convert."""
    f_min = max(fD[0].min(), rD[0].min())
    f_max = min(fD[0].max(), rD[0].max())
    df = max(fD[0][1] - fD[0][0], rD[0][1] - rD[0][0])
    fg = np.arange(f_min, f_max + df / 2, df)
    ph = (np.interp(fg, fD[0], fD[1]) + np.interp(fg, rD[0], rD[1])) / 2
    ch = np.minimum(np.interp(fg, fD[0], fD[2]),
                    np.interp(fg, rD[0], rD[2]))
    return phase_to_dispersion(fg, ph, ch, spacing, P)


def collect_dispersion(data_dir, tests, P, log=print):
    if P.get("qc_mode", "legacy") == "consensus":
        return collect_dispersion_consensus(data_dir, tests, P, log)
    return collect_dispersion_legacy(data_dir, tests, P, log)


def collect_dispersion_legacy(data_dir, tests, P, log=print):
    """Run the full acceptance hierarchy. Returns (wl, vph, spacing, trust)
    arrays plus a QC report list."""
    report = []

    def _log(msg):
        report.append(msg)
        log("    " + msg)

    groups = {}
    for folder, spacing, direction in tests:
        try:
            freq, real, imag, coh = read_record(data_dir, folder)
            if P.get("unwrap_mode", "legacy") == "segmented":
                phase = process_phase_segmented(freq, real, imag, coh,
                                                spacing, P)
            else:
                phase = process_phase(freq, real, imag)
            groups.setdefault(spacing, {}).setdefault(direction, []).append(
                (freq, phase, coh, folder))
        except Exception as e:
            _log(f"{folder}: read error {e}")

    def disp(rec, spacing, off=0.0):
        return phase_to_dispersion(rec[0], rec[1] + off, rec[2], spacing, P)

    # per-record gate; several records may exist per (spacing, direction)
    # (e.g. hammer + drop-weight) — keep the best-quality passing one
    valid = {}
    for spacing, dirs in sorted(groups.items()):
        for direction, recs in dirs.items():
            best, best_key = None, None
            for rec in recs:
                wl, vph, f = disp(rec, spacing)
                n, rho = record_quality(wl, vph, f, P)
                if n >= P["min_pts"] and rho >= P["spearman_min"]:
                    key = (rho, n)
                    if best is None or key > best_key:
                        best, best_key = rec, key
                elif n >= P["min_pts"]:
                    _log(f"{rec[3]}: REJECTED, no coherent phase trend "
                         f"(rho={rho:+.2f}, n={n})")
                else:
                    _log(f"{rec[3]}: REJECTED, only {n} valid points")
            if best is not None:
                valid.setdefault(spacing, {})[direction] = best
                if len(recs) > 1:
                    _log(f"d={spacing:g}{direction}: using {best[3]} "
                         f"(best quality of {len(recs)} records)")

    # pass 1: trusted spacings (F/R present and agreeing at zero offset)
    trusted, pending = [], []
    for spacing, dirs in sorted(valid.items()):
        if len(dirs) == 2:
            wF, vF, _ = disp(dirs['F'], spacing)
            wR, vR, _ = disp(dirs['R'], spacing)
            dis = fr_disagreement(wF, vF, wR, vR)
            if dis <= P["fr_agree"]:
                wl, vph, _ = average_pair(dirs['F'], dirs['R'], spacing, P)
                trusted.append((spacing, wl, vph, P["trust_avg"],
                                f"F/R avg (agree {dis*100:.0f}%)"))
                continue
        pending.append((spacing, dirs))

    def tdev(wl, vph):
        ds = [overlap_dev(wl, vph, t[1], t[2]) for t in trusted]
        ds = [d for d in ds if np.isfinite(d)]
        return min(ds) if ds else np.nan

    # pass 2: salvage / single-direction resolution
    accepted = list(trusted)
    R = P["offset_range"]
    for spacing, dirs in pending:
        if len(dirs) == 2:
            best = None
            for oF, oR in product(range(-R, R + 1), repeat=2):
                wF, vF, _ = disp(dirs['F'], spacing, oF * 2 * np.pi)
                wR, vR, _ = disp(dirs['R'], spacing, oR * 2 * np.pi)
                if len(wF) < P["min_pts"] or len(wR) < P["min_pts"]:
                    continue
                dis = fr_disagreement(wF, vF, wR, vR)
                if dis > P["fr_agree"]:
                    continue
                wl = np.concatenate([wF, wR])
                vph = np.concatenate([vF, vR])
                dev = tdev(wl, vph)
                if np.isfinite(dev) and dev > P["cross_dev"]:
                    continue
                score = (dev if np.isfinite(dev) else 0.5,
                         abs(oF) + abs(oR), dis)
                if best is None or score < best[0]:
                    best = (score, oF, oR, wl, vph, dis, dev)
            if best is not None:
                _, oF, oR, wl, vph, dis, dev = best
                dtxt = f"{dev*100:.0f}%" if np.isfinite(dev) else "no overlap"
                trust = (P["trust_validated"] if np.isfinite(dev)
                         else P["trust_unverified"])
                accepted.append((spacing, wl, vph, trust,
                                 f"SALVAGED offsets F{oF:+d}/R{oR:+d}*2pi "
                                 f"(agree {dis*100:.0f}%, dev {dtxt})"))
                _log(f"d={spacing:g}: salvaged with offsets "
                     f"F{oF:+d}/R{oR:+d}*2pi (F/R {dis*100:.0f}%, "
                     f"cross-dev {dtxt})")
                continue
            cand = {}
            for direction, rec in dirs.items():
                wl, vph, _ = disp(rec, spacing)
                cand[direction] = (wl, vph, tdev(wl, vph))
            ok = [d for d, (_, _, v) in cand.items()
                  if np.isfinite(v) and v <= P["cross_dev"]]
            if len(ok) == 1:
                wl, vph, dev = cand[ok[0]]
                accepted.append((spacing, wl, vph, P["trust_validated"],
                                 f"{ok[0]} kept (dev {dev*100:.0f}%)"))
                _log(f"d={spacing:g}: F/R disagree, kept {ok[0]} "
                     f"(cross-validated {dev*100:.0f}%)")
            else:
                _log(f"d={spacing:g}: REJECTED (F/R disagree, no salvage, "
                     f"no cross-validation)")
        else:
            direction, rec = next(iter(dirs.items()))
            wl, vph, _ = disp(rec, spacing)
            dev = tdev(wl, vph)
            if np.isfinite(dev) and dev > P["cross_dev"]:
                _log(f"{rec[3]}: REJECTED, contradicts trusted spacing "
                     f"by {dev*100:.0f}%")
                continue
            accepted.append((spacing, wl, vph, P["trust_unverified"],
                             f"{rec[3]} ({direction} only)"))

    # conservative mode: restrict non-trusted data to the wavelength range
    # covered by trusted spacings (deep Vs from cross-validated data only)
    if P.get("conservative_deep") and trusted:
        t_lmax = max(t[1].max() for t in trusted)
        restricted = []
        for spacing, wl, vph, trust, label in accepted:
            if trust > P["trust_avg"]:
                m = wl <= t_lmax
                if m.sum() < P["min_pts"]:
                    _log(f"d={spacing:g}: dropped in conservative mode "
                         f"(no points within validated range, "
                         f"lambda <= {t_lmax:.1f} m)")
                    continue
                if m.sum() < len(wl):
                    _log(f"d={spacing:g}: trimmed to validated range "
                         f"(lambda <= {t_lmax:.1f} m, "
                         f"{len(wl) - m.sum()} pts removed)")
                wl, vph = wl[m], vph[m]
            restricted.append((spacing, wl, vph, trust, label))
        accepted = restricted

    # pass 3: continuity gate for unverified records
    def edge_step(wlA, vA, wlB, vB):
        if np.median(wlA) < np.median(wlB):
            iA = np.argsort(wlA)[-max(3, len(wlA) // 4):]
            iB = np.argsort(wlB)[:max(3, len(wlB) // 4)]
        else:
            iA = np.argsort(wlA)[:max(3, len(wlA) // 4)]
            iB = np.argsort(wlB)[-max(3, len(wlB) // 4):]
        return abs(np.median(vA[iA]) - np.median(vB[iB])) / \
            np.median(vB[iB])

    kept = [a for a in accepted if a[3] < P["trust_unverified"]]
    for spacing, wl, vph, trust, label in sorted(
            [a for a in accepted if a[3] >= P["trust_unverified"]],
            key=lambda x: np.median(x[1])):
        if kept:
            steps = [edge_step(wl, vph, w2, v2)
                     for _, w2, v2, _, _ in kept]
            if min(steps) > P["continuity_step"]:
                _log(f"d={spacing:g}: REJECTED by continuity "
                     f"({min(steps)*100:.0f}% step at join — likely "
                     f"body-wave/higher-mode contamination)")
                continue
        kept.append((spacing, wl, vph, trust, label))

    kept.sort(key=lambda x: x[0])
    for spacing, wl, vph, trust, label in kept:
        _log(f"d={spacing:>4g} m: {len(wl):4d} pts, "
             f"lambda [{wl.min():5.1f}-{wl.max():5.1f}] m, "
             f"Vph [{np.min(vph):4.0f}-{np.max(vph):4.0f}] m/s, "
             f"trust x{trust:.1f}  [{label}]")
    if not kept:
        raise RuntimeError("all records rejected by QC")
    wl = np.concatenate([k[1] for k in kept])
    vph = np.concatenate([k[2] for k in kept])
    sp = np.concatenate([np.full(len(k[1]), k[0]) for k in kept])
    tr = np.concatenate([np.full(len(k[1]), k[3]) for k in kept])
    return wl, vph, sp, tr, report


def collect_dispersion_consensus(data_dir, tests, P, log=print):
    """Revised acceptance: F/R averaging + cross-spacing consensus.

    Field evidence (164 sites): forward and reverse phases agree at
    lambda/d >= 2 but drift apart smoothly as frequency rises, the
    signature of an antisymmetric receiver/coupling phase error that the
    F/R average is designed to cancel (Nazarian & Stokoe 1984). The legacy
    20 % F/R gate therefore discards records the average can repair, and
    sites with no passing pair end up with *unrestricted* deep data.

    Here every spacing contributes one candidate curve: the F/R phase
    average of the best-agreeing run pair (or the single direction when
    only one passes Gate 1). Candidates are linked when their curves agree
    within cross_dev over their common wavelength range; the largest
    connected group (ties: lowest mean F/R disagreement) is accepted.
    Trust: 1.0 if F/R agree within fr_agree, 1.5 if averaged but
    disagreeing, 2.0 if single-direction. A site whose group has a single
    member is flagged 'unvalidated'. z_res follows from the accepted
    group only, so depth is never set by data no other spacing confirms.
    """
    report = []

    def _log(msg):
        report.append(msg)
        log("    " + msg)

    resolved = None
    if P.get("unwrap_mode", "legacy") == "chain":
        from .chain import chain_phases
        resolved = chain_phases(data_dir, tests, P, _log)
    groups = {}
    for folder, spacing, direction in tests:
        try:
            if resolved is not None:
                if folder not in resolved:
                    _log(f"{folder}: REJECTED, no coherent segment")
                    continue
                freq, phase, coh = resolved[folder]
            else:
                freq, real, imag, coh = read_record(data_dir, folder)
                phase = process_phase(freq, real, imag)
            wl, vph, f = phase_to_dispersion(freq, phase, coh, spacing, P)
            n, rho = record_quality(wl, vph, f, P)
            if n >= P["min_pts"] and rho >= P["spearman_min"]:
                groups.setdefault(spacing, {}).setdefault(direction, []).append(
                    ((freq, phase, coh, folder), wl, vph, rho, n))
            else:
                _log(f"{folder}: REJECTED gate 1 (n={n}, rho={rho:+.2f})")
        except Exception as e:
            _log(f"{folder}: read error {e}")

    cands = {}
    for spacing, dirs in sorted(groups.items()):
        if "F" in dirs and "R" in dirs:
            best = None
            for a in dirs["F"]:
                for b in dirs["R"]:
                    dis = fr_disagreement(a[1], a[2], b[1], b[2])
                    if best is None or dis < best[0]:
                        best = (dis, a, b)
            dis, a, b = best
            wl, vph, _ = average_pair(a[0], b[0], spacing, P)
            if len(wl) >= P["min_pts"]:
                trust = (P["trust_avg"] if dis <= P["fr_agree"]
                         else P["trust_validated"])
                cands[spacing] = (wl, vph, trust, dis,
                                  f"F/R avg {a[0][3]}+{b[0][3]} "
                                  f"(F/R {dis*100:.0f}%)")
                continue
        recs = [r for d in dirs.values() for r in d]
        r = max(recs, key=lambda x: (x[3], x[4]))
        cands[spacing] = (r[1], r[2], P["trust_unverified"], np.nan,
                          f"{r[0][3]} single direction")

    sp = sorted(cands)
    adj = {s: set() for s in sp}
    for i, s1 in enumerate(sp):
        for s2 in sp[i + 1:]:
            dev = overlap_dev(cands[s1][0], cands[s1][1],
                              cands[s2][0], cands[s2][1])
            if np.isfinite(dev) and dev <= P["cross_dev"]:
                adj[s1].add(s2)
                adj[s2].add(s1)
    comps, seen = [], set()
    for s in sp:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(sorted(comp))

    def key(c):
        dis = [cands[s][3] for s in c if np.isfinite(cands[s][3])]
        return (len(c), -(np.mean(dis) if dis else 9.0))
    if not comps:
        raise RuntimeError("all records rejected by QC")
    best = max(comps, key=key)
    if len(best) == 1:
        # no two spacings agree: keep the best-agreeing F/R pair alone
        best = [min(sp, key=lambda s: (np.nan_to_num(cands[s][3], nan=9.0),
                                        cands[s][2]))]
        _log(f"UNVALIDATED site: no two spacings agree; kept d={best[0]:g}")
    for s in sp:
        if s not in best:
            _log(f"d={s:g}: REJECTED, outside consensus group "
                 f"[{cands[s][4]}]")
    kept = [(s, cands[s][0], cands[s][1], cands[s][2], cands[s][4])
            for s in best]
    for s, wl, vph, trust, label in kept:
        _log(f"d={s:>4g} m: {len(wl):4d} pts, lambda [{wl.min():5.1f}-"
             f"{wl.max():5.1f}] m, trust x{trust:.1f}  [{label}]")
    wl = np.concatenate([k[1] for k in kept])
    vph = np.concatenate([k[2] for k in kept])
    spa = np.concatenate([np.full(len(k[1]), k[0]) for k in kept])
    tr = np.concatenate([np.full(len(k[1]), k[3]) for k in kept])
    return wl, vph, spa, tr, report


def coherent_share(data_dir, tests, P):
    """Record-quality indicator: the mean, over all records of a site, of
    the fraction of the spectrum above 5 Hz whose coherence passes the mask
    threshold. It uses the raw coherence only (no phase, no inversion).

    On the 164-site dataset its quartiles are 0.30 / 0.36 / 0.43, and the
    lowest quartile is the only site-quality indicator, of some twenty
    tested, that predicts poorer agreement with independent penetration
    data (permutation p = 0.001-0.002 for removing the 10-13 least coherent
    of 51 soundings). P["coh_share_min"] (default 0.30) flags a site as
    low-coherence; the flag is advisory and does not change the profile."""
    from .phase import coherence_threshold
    vals = []
    for folder, spacing, _ in tests:
        try:
            freq, _, _, coh = read_record(data_dir, folder)
        except Exception:
            continue
        vals.append(float(np.mean(coh[freq > 5] >=
                                  coherence_threshold(spacing, P))))
    share = float(np.mean(vals)) if vals else float("nan")
    return share, bool(share < P.get("coh_share_min", 0.30))
