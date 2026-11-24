"""Paths and existing solver-free fitting and validation routines."""
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"core15_window_selection_2026-11-18"
sys.path.insert(0,str(PRIOR))
from setup_study import E, TrainingMoments, corrected_fit, np, records, score
from evaluate import matrices
from benchmark import stability
import train
sys.path.insert(0,str(ROOT))

SEEDS=range(794001,794065)
