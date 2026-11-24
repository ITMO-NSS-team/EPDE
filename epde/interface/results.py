"""Portable discovery output. No member retains a cache, optimizer, or training tensor."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


def _encode_number(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'$float': str(value)}
    return value


def _map_numbers(value, decode=False):
    if isinstance(value, dict):
        if decode and value.keys() == {'$float'}:
            return float(value['$float'])
        return {k: _map_numbers(v, decode) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_numbers(v, decode) for v in value]
    return value if decode else _encode_number(value)


def _term(term):
    return [dict(label=f.label, family=f.ftype, derivative=_json_value(f.deriv_code),
                 parameters={spec['name']: _json_value(f.params[i])
                             for i, spec in f.params_description.items()})
            for f in term.structure]


def _equation(eq):
    fitted = eq.weights_final_evald
    target = eq.target_idx if fitted else None
    terms = [dict(factors=_term(term), coefficient=(
        float(eq.weights_final[eq.weight_index(i, target)]) if fitted else None))
             for i, term in enumerate(eq.structure) if i != target]
    return dict(variable=eq.main_var_to_explain, text=eq.text_form,
                target=_term(eq.target) if fitted else None, terms=terms,
                intercept=float(eq.intercept) if fitted else None)


def _system(system):
    return dict(equations=[_equation(eq) for eq in system.vals],
                objectives=np.atleast_1d(system.obj_fun).astype(float).tolist())


@dataclass
class DiscoveryResult:
    mode: str
    levels: list
    ranked: list
    variable_fronts: dict
    measurements: list

    @classmethod
    def from_search(cls, search):
        population = search._resulting_population
        levels, ranked, fronts = [], [], {}
        if search.multiobjective_mode:
            from epde.optimizers.nsga.per_variable import variable_fronts
            levels = [[_system(system) for system in level] for level in population]
            systems = (search.optimizer.reported_front() if hasattr(search.optimizer, 'reported_front')
                       else [s for level in population for s in level])
            for key, items in variable_fronts(systems).items():
                fronts[key] = [dict(equation=_equation(eq), objectives=list(values)) for eq, values in items]
        else:
            ranked = [_system(system) for system in population]
        measurements = [dict(trajectory=_json_value(key), variable=variable, **_json_value(info))
                        for (key, variable), info in search.measurement_summary.items()]
        return cls('multiobjective' if search.multiobjective_mode else 'single', levels, ranked,
                   fronts, measurements)

    @property
    def front(self):
        """First Pareto level or the best ranked single-objective candidate."""
        if self.mode == 'single':
            return self.ranked[:1]
        return self.levels[0] if self.levels else []

    def as_dict(self):
        return dict(schema_version=1, **_map_numbers(asdict(self)))

    @classmethod
    def from_dict(cls, record):
        if record.get('schema_version') != 1:
            raise ValueError('Unsupported EPDE result schema')
        return cls(**{k: _map_numbers(record[k], decode=True) for k in cls.__dataclass_fields__})

    def save(self, path):
        Path(path).write_text(json.dumps(self.as_dict(), allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
