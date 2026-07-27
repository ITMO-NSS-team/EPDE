"""Data adapter for the REAL TR-PIV cylinder wake (unknown-truth config).

See configs/cylinder_piv.yaml. Von Karman Institute / MODULO Ex_5
time-resolved PIV: 71x30 spatial grid (mm units in MESH.dat), velocity
snapshots ``Res{i:05d}.dat`` (m/s) at ~3 kHz. Loading conventions mirror
``projects/pic/data/cylinder_piv/signal_check.py``: NaN gaps zero-filled,
then a light Gaussian denoise BEFORE any differentiation (sigma~1 is the
established sweet spot; heavier blur collapses the shedding structure).

Knobs (YAML ``adapter_kwargs`` / run.py ``--adapter-kwarg``):
    denoise_sigma  Gaussian sigma applied on (t, y, x); 0 disables.
    i0, i1         contiguous snapshot block (developed shedding: 300..800).
    decimate_t     keep every k-th snapshot (halves load time and series).
"""

import os
import zipfile
from io import StringIO

import numpy as np
from scipy.ndimage import gaussian_filter

_PIC_PIV = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'cylinder_piv'
))
_ZIP = os.path.join(_PIC_PIV, 'Ex_5_TR_PIV_Cylinder.zip')
_DT = 1.0 / 3000.0      # TR-PIV ~3 kHz


def _load_dat(z, name):
    return np.loadtxt(StringIO(z.read(name).decode('latin-1')), skiprows=1)


def load_data(denoise_sigma=1.0, i0=300, i1=800, decimate_t=1):
    z = zipfile.ZipFile(_ZIP)
    mesh = _load_dat(z, 'MESH.dat')
    x_mm, y_mm = mesh[:, 0], mesh[:, 1]
    n_pts = mesh.shape[0]
    nx = int(np.where(np.diff(y_mm) != 0)[0][0]) + 1    # x is the fast axis
    ny = n_pts // nx
    x = (x_mm.reshape(ny, nx))[0, :] * 1e-3             # meters
    y = (y_mm.reshape(ny, nx))[:, 0] * 1e-3

    u_frames, v_frames = [], []
    for i in range(i0, i1, decimate_t):
        a = _load_dat(z, f'Res{i:05d}.dat')
        u_frames.append(a[:, 0].reshape(ny, nx))
        v_frames.append(a[:, 1].reshape(ny, nx))
    u = np.nan_to_num(np.array(u_frames))
    v = np.nan_to_num(np.array(v_frames))
    if denoise_sigma and denoise_sigma > 0:
        sig = (float(denoise_sigma),) * 3
        u = gaussian_filter(u, sig)
        v = gaussian_filter(v, sig)

    t = np.arange(u.shape[0], dtype=np.float64) * (_DT * decimate_t)
    grids = np.meshgrid(t, y, x, indexing='ij')
    return tuple(grids), [u, v], ['u', 'v'], 2
