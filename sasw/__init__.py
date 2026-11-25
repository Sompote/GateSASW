"""sasw — quality-controlled SASW processing: raw two-receiver spectra ->
dispersion curve -> Vs profile, with optional SPT-constrained joint
inversion. See run_sasw.py for the command-line entry point."""

__version__ = "2.0.0"

from .config import DEFAULTS
from .io import auto_detect_tests, read_record, read_spt_csv
from .qc import collect_dispersion, coherent_share
from .composite import build_composite
from .forward import forward_vph
from .inversion import (invert, joint_inversion_sweep, profile_at,
                        resolution_depth, resolution_window)
from .correlations import EQUATIONS, select_best, build_priors
