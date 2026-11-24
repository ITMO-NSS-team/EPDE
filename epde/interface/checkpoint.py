"""Trusted, local epoch checkpoints for deterministic built-in CPU searches."""
import hashlib
import json
import os
import pickle
import platform
import tempfile
from importlib import metadata
from pathlib import Path
from zipfile import ZipFile

import cloudpickle
import numpy as np
import torch

import epde.globals as g
from epde.interface.runtime import random_state, restore_random_state
from epde.interface.search_config import active_config
from epde.operators.common import fill_prune, plain_fit


def _value(value):
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(value)
        return [data.dtype.str, list(data.shape), hashlib.sha256(data.view(np.uint8).reshape(-1)).hexdigest()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    if callable(value):
        module = getattr(value, '__module__', '')
        if module.startswith(('epde.', 'numpy', 'sklearn.')):
            return module + '.' + value.__qualname__
        raise ValueError(f'Checkpoint requires a built-in implementation, got {value!r}')
    return value


def _manifest(search, init_params):
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    data = []
    for key, traj in g.samples_manager._traj.items():
        specification = getattr(traj._preprocessor_pipeline, 'specification', None)
        data.append([key, getattr(traj, 'axis_names', None), specification, traj._domain._time_axis,
                     traj.variable_names, traj.grids(), traj._domain.g_func,
                     [traj.raw_field(name) for name in traj.variable_names], traj._derivs, traj.raw_tokens(),
                     [getattr(entry, 'derivatives', None) for entry in traj._entries]])
    config = active_config().as_dict()
    config['evolution'].pop('training_epochs')
    initialization = {k: v for k, v in init_params.items() if k != 'population_instruct'}
    structure = dict(init_params['population_instruct'])
    structure.pop('pool')
    families = [[f.ftype, f.tokens, f.token_params, f.status, f.equality_ranges, getattr(f, 'param_grids', {})]
                for f in search.pool.families]
    return _value(dict(version=1, source=digest.hexdigest(), python=platform.python_version(),
                       platform=platform.platform(), libraries={n: metadata.version(n) for n in
                       ('cloudpickle', 'numpy', 'scipy', 'scikit-learn', 'torch')},
                       config=config, initialization=initialization, structure=structure,
                       patience=search.optimizer_exec_params.get('patience'), data=data, families=families))


class _Writer(cloudpickle.CloudPickler):
    def persistent_id(self, obj):
        return self.references.get(id(obj))


class _Reader(pickle.Unpickler):
    def persistent_load(self, key):
        return self.references[key]


class EpochCheckpoint:
    def __init__(self, path, resume=False, total_epochs=None):
        self.path = Path(path)
        self.resume = resume
        self.total_epochs = total_epochs
        self.completed = 0
        self.stopped = False
        if total_epochs is not None and (type(total_epochs) is not int or total_epochs < 1):
            raise ValueError('total_epochs must be a positive integer')

    @staticmethod
    def validate_preprocessor(trajectory):
        if trajectory._derivs is not None:
            return
        specification = getattr(trajectory._preprocessor_pipeline, 'specification', None)
        if specification is None:
            raise ValueError('Checkpoint requires a built-in preprocessor or supplied derivatives')
        kind, kwargs, _ = specification
        if kind == 'ANN':
            from epde.preprocessing.smoothers import ANNSmoother
            if ANNSmoother.resolve_device(kwargs.get('device', 'auto')).type != 'cpu':
                raise ValueError("Epoch checkpoints require CPU ANN preprocessing; set preprocessor_kwargs={'device': 'cpu'}")

    def begin(self):
        if self.resume:
            with ZipFile(self.path) as archive:
                py, numpy, tensor = json.loads(archive.read('preparation.json'))
                self.preparation_rng = ((py[0], tuple(py[1]), py[2]),
                                        (numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]),
                                        torch.tensor(tensor, dtype=torch.uint8, device='cpu'))
            restore_random_state(self.preparation_rng)
        else:
            self.preparation_rng = random_state()

    def prepare(self, search, init_params):
        self.manifest = _manifest(search, init_params)
        self.refs = {('pool',): search.pool, ('search',): search, ('manager',): g.samples_manager}
        for i, family in enumerate(search.pool.families):
            self.refs['family', i] = family
        for key, traj in g.samples_manager._traj.items():
            self.refs['trajectory', key] = traj
            self.refs['domain', key] = traj._domain
            for i, entry in enumerate(traj._entries):
                self.refs['entry', key, i] = entry
        # Base tensors are rebuilt from verified data. Cached products that
        # equations mark as saved share the pickle graph with the population.
        for key, table in g.tensor_cache.memory_default.items():
            for label, array in table.items():
                self.refs['base', key, label] = array
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume:
            with ZipFile(self.path) as archive:
                if json.loads(archive.read('manifest.json')) != self.manifest:
                    raise ValueError('Checkpoint data, settings or software differ from this search')
                with archive.open('state.pkl') as stream:
                    reader = _Reader(stream)
                    reader.references = self.refs
                    state = reader.load()
            self.completed, self.stopped = state['epoch'], state['stopped']
            search.optimizer = state['optimizer']
            search._stagnation = state['stagnation']
            plain_fit.restore_checkpoint_state(state['plain'])
            fill_prune.restore_checkpoint_state(state['fill_prune'])
            g.tensor_cache.memory_default.update(state['tensors'])
            g.tensor_cache.memory_normalized.update(state['normalized'])
            if g.samples_manager.evaluation_rows is not None:
                vars(g.samples_manager.evaluation_rows).update(state['row_state'])
            g.history = state['history']
            g.vc_modes_cache.update(state['modes'])
            restore_random_state(state['rng'])

    def save(self, search, epoch, stopped):
        rows = g.samples_manager.evaluation_rows
        state = dict(epoch=int(epoch), stopped=bool(stopped), optimizer=search.optimizer,
                     stagnation=search._stagnation, history=g.history, modes=g.vc_modes_cache,
                     tensors=g.tensor_cache.memory_default, normalized=g.tensor_cache.memory_normalized,
                     rng=random_state(), plain=plain_fit.checkpoint_state(), fill_prune=fill_prune.checkpoint_state(),
                     row_state={k: getattr(rows, k) for k in ('_columns', '_last_frequency_column',
                                'frequency_fits', 'frequency_solves', '_observed_frequencies') if rows is not None})
        with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as file:
            tmp = Path(file.name)
        try:
            with ZipFile(tmp, 'w') as archive:
                py, numpy, tensor = self.preparation_rng
                archive.writestr('preparation.json', json.dumps(
                    [py, [numpy[0], numpy[1].tolist(), *numpy[2:]], tensor.tolist()]))
                archive.writestr('manifest.json', json.dumps(self.manifest, allow_nan=False))
                with archive.open('state.pkl', 'w', force_zip64=True) as stream:
                    writer = _Writer(stream, protocol=pickle.HIGHEST_PROTOCOL)
                    writer.references = {id(v): k for k, v in self.refs.items()}
                    writer.dump(state)
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
