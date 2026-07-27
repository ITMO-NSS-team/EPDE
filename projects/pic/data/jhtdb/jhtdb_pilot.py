"""JHTDB pilot slab download: forced isotropic turbulence
(isotropic1024coarse), one x-y plane, velocity + pressure + server-side
gradients/Laplacians, small time window.

Purpose (gate protocol): verify the full NS momentum balance
  du/dt + (u.grad)u = -grad p + nu*lap u
closes on the queried plane (R^2 ~ 1) before any filtering/EPDE wiring.
Server-side spatial operators make every spatial term DNS-exact; only
du/dt is frame-differenced (stored frame spacing dt = 0.002).

Public testing token: fine at this scale (~200 queries x 2304 points).
"""
import os
import time as _time

import numpy as np
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

TOKEN = 'edu.jhu.pha.turbulence.testing-201406'
DATASET = 'isotropic1024coarse'
NU = 0.000185

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(HERE, 'jhtdb_pilot_plane.npz')
SCRATCH = os.path.join(HERE, '_giverny_tmp')

NX = NY = 48
STRIDE = 8                      # grid spacing in DNS cells
DNS_DX = 2 * np.pi / 1024
DX = DNS_DX * STRIDE
# query points snapped to exact DNS grid nodes so the no-interpolation FD
# operators (the only ones the laplacian supports) are exact and consistent
# with the field values
IX0 = IY0 = 163
IZ0 = 512
T0, NT, DT = 0.5, 40, 0.002

# spatial method is operator-specific: lag8 is field-only; laplacian only
# takes fd*; at grid nodes fd8noint is exact 8th-order FD on the DNS grid.
QUERIES = (('velocity', 'field', 'lag8', 3), ('pressure', 'field', 'lag8', 1),
           ('velocity', 'gradient', 'fd8noint', 9),
           ('pressure', 'gradient', 'fd8noint', 3),
           ('velocity', 'laplacian', 'fd8noint', 3))


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    dataset = turb_dataset(dataset_title=DATASET, output_path=SCRATCH,
                           auth_token=TOKEN)
    xs = DNS_DX * (IX0 + STRIDE * np.arange(NX))
    ys = DNS_DX * (IY0 + STRIDE * np.arange(NY))
    z0 = DNS_DX * IZ0
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    points = np.column_stack([XX.ravel(), YY.ravel(),
                              np.full(XX.size, z0)])

    store = {f'{var}_{op}': np.empty((NT, NY * NX, ncomp), dtype=np.float32)
             for var, op, _method, ncomp in QUERIES}
    times = T0 + DT * np.arange(NT)

    t_start = _time.time()
    for it, t in enumerate(times):
        for var, op, method, ncomp in QUERIES:
            for attempt in range(4):
                try:
                    res = getData(dataset, var, float(t), 'none', method,
                                  op, points)
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise
                    print(f'retry {attempt + 1} after error: {exc}')
                    _time.sleep(5.0 * (attempt + 1))
            arr = np.asarray(res, dtype=np.float32).reshape(NY * NX, ncomp)
            store[f'{var}_{op}'][it] = arr
        if it % 5 == 0 or it == NT - 1:
            el = _time.time() - t_start
            print(f'frame {it + 1}/{NT} done, elapsed {el:.0f}s', flush=True)

    np.savez_compressed(
        OUT_NPZ,
        x=xs.astype(np.float64), y=ys.astype(np.float64), z=np.float64(z0),
        t=times.astype(np.float64), nu=np.float64(NU), stride=STRIDE,
        **{k: v.reshape(NT, NY, NX, v.shape[-1]) for k, v in store.items()})
    print('saved', OUT_NPZ)


if __name__ == '__main__':
    main()
