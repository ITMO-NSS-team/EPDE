"""Data adapter for the REAL Hudson's Bay lynx-hare records (Lotka-Volterra).

See configs/lv_real.yaml. Wraps the established loader in
``projects/pic/data/lv/lv_real.py`` (cubic-spline densification of the 21
annual pelt records) rather than duplicating it. ``densify`` is exposed as
an adapter kwarg (YAML ``adapter_kwargs`` / run.py ``--adapter-kwarg``).
"""

import importlib.util
import os

_PIC_LV = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'lv'
))


def _pic_lv_real():
    # Load pic's lv_real.py under a private name: a bare ``import lv_real``
    # would collide with THIS module (adapters/lv_real.py) on sys.path.
    spec = importlib.util.spec_from_file_location(
        'pic_lv_real', os.path.join(_PIC_LV, 'lv_real.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(densify=10):
    t, hare, lynx = _pic_lv_real().load_lynxhare(densify=densify)
    return (t,), [hare, lynx], ['u', 'v'], 0
