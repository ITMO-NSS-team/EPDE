"""Paired tracks for collision discovery with observed contact states and projected kinematics."""
import json
from pathlib import Path

import numpy as np

from physics import CAP, FPS, OUT, arena_for, fit_self, swarm_files, swarm_meta
from preprocess_tracks import load_product


def projected_tracks():
    observations, tracks = {}, {}
    sources = json.loads((OUT/'inputs.json').read_text())
    for source in swarm_files():
        tag = swarm_meta(source)['tag']
        path = OUT/'preprocessed'/f'{tag}.npz'
        with np.load(path) as z:
            assert str(z['raw_sha256']) == sources[Path(source).name]
            assert str(z['recording']) == tag and z['fps'] == FPS and z['cap_px'] == CAP
        observed, corrected, _ = load_product(path)
        observations[tag], tracks[tag] = observed, corrected
    assert len(tracks) == 5
    return observations, tracks


def fold_calibrations(tracks):
    params = {}
    for held in tracks:
        train = [tr.part(0,3*len(tr.pos)//4) for tag,tr in tracks.items() if tag != held]
        arenas = arena_for(train)
        params[held] = (*fit_self(train,arenas),arenas)
    return params
