"""Nov 8 (wire U): log the global random stream -- every np.random.* / random.* call with its value and the calling
frames -- to <PREFIX>_rng.log, the first TRACE_RNG calls. Read-only on the search: the wrappers return what the
wrapped function returns. Frames inside numpy, random and this module are skipped; four frames are shown."""
import os
import random as pyrandom
import sys

import numpy as np

_NP = ('randint', 'random', 'random_sample', 'rand', 'randn', 'choice', 'uniform', 'normal', 'permutation',
       'shuffle', 'standard_normal', 'binomial', 'poisson', 'exponential', 'beta', 'gamma', 'integers')
_PY = ('random', 'randint', 'choice', 'choices', 'sample', 'shuffle', 'uniform', 'gauss', 'randrange')
STATE = {'n': 0, 'limit': 0, 'out': None}


def _frames():
    out = []
    f = sys._getframe(2)
    while f is not None and len(out) < 4:
        name = f.f_code.co_filename
        base = os.path.basename(name)
        if not (base == 'rng_trace.py' or os.sep + 'numpy' + os.sep in name or base == 'random.py'):
            out.append('%s:%d:%s' % (base, f.f_lineno, f.f_code.co_name))
        f = f.f_back
    return ' < '.join(out)


def _short(value):
    try:
        if isinstance(value, np.ndarray):
            if value.size <= 6:
                return 'arr%s' % np.array2string(value.ravel(), precision=6)
            return 'arr[%d]%.6g..' % (value.size, float(value.ravel()[0]))
        text = repr(value)
    except Exception:  # noqa: BLE001
        text = '<?>'
    return text if len(text) <= 60 else text[:57] + '...'


def _wrap(module, name, tag):
    real = getattr(module, name, None)
    if real is None:
        return

    def traced(*args, **kwargs):
        value = real(*args, **kwargs)
        if STATE['n'] < STATE['limit']:
            STATE['n'] += 1
            arg = _short(args[0]) if args else ''
            STATE['out'].write('%d\t%s.%s(%s)\t%s\t%s\n' % (STATE['n'], tag, name, arg, _short(value), _frames()))
            if STATE['n'] % 200 == 0:
                STATE['out'].flush()
        return value
    traced.__name__ = getattr(real, '__name__', name)
    setattr(module, name, traced)


def install(prefix, here):
    STATE['limit'] = int(os.environ.get('TRACE_RNG') or 0)
    if STATE['limit'] <= 0:
        return
    STATE['out'] = open(os.path.join(here, prefix + '_rng.log'), 'w')
    import epde
    STATE['out'].write('# epde from %s\n' % os.path.dirname(epde.__file__))
    for name in _NP:
        _wrap(np.random, name, 'np')
    for name in _PY:
        _wrap(pyrandom, name, 'py')
    import atexit
    atexit.register(lambda: STATE['out'].close())
