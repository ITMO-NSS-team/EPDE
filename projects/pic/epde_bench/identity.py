"""Portable content identity for runs, campaign resume and notebook caches.

No Git revision participates: dirty computation edits invalidate results while
changes to documentation do not. Input files are streamed, never deserialized.
"""
import ast
import hashlib
import inspect
import json
import platform
from importlib import metadata
from pathlib import Path
from threading import Lock

from .config import load_config
from .datasets import REGISTRY
from .paths import DATA_DIR, PIC_DIR, REPO_ROOT

_HASH_CACHE = {}
_HASH_LOCK = Lock()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _file_hash(path, source=False):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
           getattr(stat, 'st_ino', 0), source)
    with _HASH_LOCK:
        if key in _HASH_CACHE:
            return _HASH_CACHE[key]
    h = hashlib.sha256()
    if source:
        h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    else:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
    result = h.hexdigest()
    with _HASH_LOCK:
        _HASH_CACHE[key] = result
    return result


def _source_hash():
    files = {}
    for folder in (REPO_ROOT / 'epde', PIC_DIR / 'epde_bench'):
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' not in path.parts:
                files[f'{folder.name}/{path.relative_to(folder).as_posix()}'] = _file_hash(path, True)
    for name in ('bench.py',):
        path = PIC_DIR / name
        if path.is_file():
            files[name] = _file_hash(path, True)
    return _digest(files)


def _data_hash(dataset):
    # Registry patterns describe the actual inputs, not the entire data tree
    # (which also contains large search checkpoints and pickled results).
    patterns = [item.strip().split(' (', 1)[0] for item in REGISTRY[dataset].files.split(',')]
    # Include additional literal inputs used by loaders (e.g. laser.npy),
    # even if the human-facing registry summary names only the main archive.
    tree = ast.parse(inspect.getsource(REGISTRY[dataset].loader))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'data_path':
            if node.args and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                                 for arg in node.args):
                patterns.append('/'.join(arg.value for arg in node.args))
    files = {}
    for pattern in sorted(set(patterns)):
        matches = sorted(DATA_DIR.glob(pattern))
        if not matches:
            files[pattern] = None  # missing inputs are part of the experiment too
        for path in matches:
            if path.is_file():
                files[path.relative_to(DATA_DIR).as_posix()] = _file_hash(path)
    return _digest(files)


def _dependency_versions():
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'scipy', 'pandas', 'sympy', 'torch', 'pysindy', 'scikit-learn',
                 'PyYAML', 'psutil', 'epde', 'netCDF4', 'h5py', 'numba', 'tedeous'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def experiment_identity(dataset, variant, noise, seed, overrides=None, config_path=None):
    """Return a JSON-safe identity, including effective config and content hashes."""
    value = {'schema': 1, 'job': {'dataset': dataset, 'variant': variant,
                                'noise': float(noise), 'seed': int(seed)},
             'config': load_config(dataset, variant, overrides, config_path),
             'source_sha256': _source_hash(), 'data_sha256': _data_hash(dataset),
             'dependencies': _dependency_versions()}
    # YAML can contain tuples/date objects; reject unsupported values rather
    # than silently equating distinct configurations via string conversion.
    value = json.loads(json.dumps(value, allow_nan=False))
    value['sha256'] = _digest(value)
    return value


def problem_metadata(dataset):
    """Registry metadata without loading arrays or preparing derivatives.

    Truth declarations are detected in the loader source. This covers measured
    datasets with truth too, without inferring truth solely from their suite.
    """
    spec = REGISTRY[dataset]
    tree = ast.parse(inspect.getsource(spec.loader))
    known = any(isinstance(node, ast.keyword) and node.arg == 'truth'
                and not (isinstance(node.value, ast.Constant) and node.value.value is None)
                for node in ast.walk(tree))
    return {'kind': spec.kind, 'truth_known': known, 'title': spec.title,
            'source': spec.source, 'suite': spec.suite}
