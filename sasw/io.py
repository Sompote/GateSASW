"""Raw data reading and test auto-detection.

Expected layout (per SASW record):
    <data_dir>/<record>.trf/Run*/ASCII/Hxysv00000.txt   (transfer function)
    <data_dir>/<record>.trf/Run*/ASCII/Cxysv00000.txt   (coherence)

Record folder names encode spacing and direction, e.g. H4mR_HM.trf
(4 m spacing, reverse shot). Pattern: H<spacing>m<F|R>...
"""
import re
import numpy as np
from pathlib import Path

RECORD_RE = re.compile(r"H(\d+(?:\.\d+)?)m([FR])", re.IGNORECASE)


def auto_detect_tests(data_dir):
    """Scan data_dir for record folders; return [(folder_name, spacing_m,
    direction)] sorted by spacing."""
    tests = []
    for p in sorted(Path(data_dir).iterdir()):
        if not p.is_dir():
            continue
        m = RECORD_RE.search(p.name)
        if m and list(p.glob("Run*")):
            tests.append((p.name, float(m.group(1)), m.group(2).upper()))
    tests.sort(key=lambda t: (t[1], t[2]))
    return tests


def read_file(data_dir, folder, filename):
    """Read a two/three-column ASCII spectral export (header ends at a line
    starting with 'Hz'). Guarantees all returned columns have equal length:
    a data line is kept only if it yields the same number of numeric fields
    as the first accepted data line."""
    runs = sorted((Path(data_dir) / folder).glob("Run*"))
    if not runs:
        raise FileNotFoundError(f"no Run* folder in {folder}")
    path = runs[0] / "ASCII" / filename
    freq, col1, col2 = [], [], []
    ncol = None            # locked to the first accepted data row's width
    with open(path) as f:
        started = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if started:
                try:
                    vals = [float(p) for p in parts[:3]]
                except ValueError:
                    continue
                if len(vals) < 2:
                    continue
                if ncol is None:
                    ncol = min(len(vals), 3)
                if len(vals) < ncol:
                    continue
                freq.append(vals[0])
                col1.append(vals[1])
                if ncol > 2:
                    col2.append(vals[2])
            if line.startswith("Hz"):
                started = True
    return (np.array(freq), np.array(col1),
            np.array(col2) if col2 else None)


def read_record(data_dir, folder):
    """Return (freq, real, imag, coherence_interpolated)."""
    freq, real, imag = read_file(data_dir, folder, "Hxysv00000.txt")
    fc, coh, _ = read_file(data_dir, folder, "Cxysv00000.txt")
    return freq, real, imag, np.interp(freq, fc, coh)


def read_spt_csv(path):
    """Read SPT data CSV with columns:
    depth_top_m, depth_bot_m, N, status   (status: actual | refusal)
    Returns [(d0, d1, N, status)]."""
    import csv
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out.append((float(r["depth_top_m"]), float(r["depth_bot_m"]),
                        float(r["N"]), r["status"].strip().lower()))
    return out
