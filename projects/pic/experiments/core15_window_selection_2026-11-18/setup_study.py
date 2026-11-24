"""Shared paths and legacy measurements for the prospective default-panel test."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "core15_implicit_correction_2026-11-18"
sys.path.insert(0, str(LEGACY))
from train import E, SYSTEMS, TrainingMoments, np, records
from evaluate import corrected_fit, score
# Experiment imports add neighboring historical directories to the search path.
sys.path.insert(0, str(LEGACY))
from summarize import front

SEEDS = range(792001, 792007)
WIDTHS = ("local", "middle", "wide")
