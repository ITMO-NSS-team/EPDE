"""The single object every loader returns and every runner consumes."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Problem classes used for grouping in tables and notebooks.
KINDS = {
    'ode': 'ODE (one equation)',
    'ode_system': 'system of ODEs',
    'pde_1d': 'PDE, 1 space dimension',
    'pde_2d': 'PDE, 2 space dimensions',
    'pde_3d': 'PDE, 3 space dimensions',
}


@dataclass
class Problem:
    """Data plus everything known about it.

    ``grids`` follows EPDE's convention: one array per axis, all of the data
    shape (``np.meshgrid(..., indexing='ij')``), time first. An ODE has a
    single 1-D time array. ``data`` maps a variable name to an array of that
    same shape.

    ``truth`` holds one equation string per variable in EPDE's text form
    (coefficients are informative only; scoring compares the set of terms).
    ``truth_alternatives`` lists other systems that are equally correct for
    this particular data set (similarity solutions, identities of a soliton
    family, ...); a discovery matching any of them counts as a success.
    ``None`` truth means the governing law is unknown (real data) and the
    run is reported without structural metrics.
    """
    name: str
    title: str
    kind: str
    source: str                                  # 'synthetic' | 'real'
    grids: Tuple[np.ndarray, ...]
    data: Dict[str, np.ndarray]
    axis_names: Tuple[str, ...]
    truth: Optional[List[str]] = None
    truth_alternatives: List[List[str]] = field(default_factory=list)
    #: Extra named fields of the data shape that enter the search as tokens:
    #: ``(token_type, {label: array}, meaningful)``. ``meaningful=True`` lets
    #: a token stand alone in a term (forcing, source); ``False`` makes it a
    #: factor only (coefficient field, coupling). The runner turns each group
    #: into an EPDE ``CacheStoredTokens`` family and the PySINDy baseline into
    #: library columns, so both methods see the same building blocks.
    token_groups: List[Tuple[str, Dict[str, np.ndarray], bool]] = field(default_factory=list)
    #: Builds tokens that need a callable (``CustomTokens``). Called with the
    #: problem after ``createDomain``.
    extra_tokens: Optional[Callable[['Problem'], list]] = None
    #: The same quantities as arrays for the baseline, when ``extra_tokens``
    #: is used: ``{label: (array, meaningful)}``.
    extra_arrays: Dict[str, Tuple[np.ndarray, bool]] = field(default_factory=dict)
    #: Pre-computed derivatives (one (n_points, n_derivs) array per variable),
    #: used instead of numerical differentiation when the data is too coarse.
    derivs: Optional[List[np.ndarray]] = None
    #: Per-axis derivative orders describing the supplied axis-major columns.
    #: Required when derivs is supplied; prevents relabeling after overrides.
    deriv_orders: Optional[List[int]] = None
    notes: str = ''
    meta: dict = field(default_factory=dict)

    @property
    def variables(self) -> List[str]:
        return list(self.data)

    @property
    def dim(self) -> int:
        """Number of SPACE axes (0 for ODEs), EPDE's ``dimensionality``."""
        return len(self.grids) - 1

    @property
    def shape(self) -> tuple:
        return next(iter(self.data.values())).shape

    @property
    def named_arrays(self) -> Dict[str, Tuple[np.ndarray, bool]]:
        """Every extra field as ``{label: (array, meaningful)}``."""
        out = dict(self.extra_arrays)
        for _, tensors, meaningful in self.token_groups:
            out.update({label: (array, meaningful) for label, array in tensors.items()})
        return out

    @property
    def truth_systems(self) -> List[List[str]]:
        """The primary truth followed by every alternative."""
        if not self.truth:
            return []
        return [list(self.truth)] + [list(alt) for alt in self.truth_alternatives]

    def noisy(self, level: float, seed: int) -> Dict[str, np.ndarray]:
        """Additive Gaussian noise of ``level`` % of each variable's std --
        the convention of the group's runners -- from a dedicated generator,
        so the noise realisation depends only on ``seed`` (the old
        ``noise_data`` drew from the global numpy state)."""
        if level < 0 or not np.isfinite(level):
            raise ValueError('noise level must be a finite nonnegative percentage')
        if level and self.derivs is not None:
            raise ValueError(f'{self.name}: artificial noise with pre-computed derivatives is unsupported; '
                             'use --noise 0 or supply a consistent noisy derivative data set')
        if not level:
            return {k: v.copy() for k, v in self.data.items()}
        rng = np.random.default_rng(seed)
        return {k: v + level * 0.01 * np.std(v) * rng.standard_normal(v.shape)
                for k, v in self.data.items()}

    def summary(self) -> str:
        lines = [f'{self.name}: {self.title}',
                 f'  class: {KINDS.get(self.kind, self.kind)}, source: {self.source}',
                 f'  axes: {", ".join(self.axis_names)}; data shape: {self.shape}; '
                 f'variables: {", ".join(self.variables)}']
        for axis, grid in zip(self.axis_names, self.grids):
            values = np.unique(np.asarray(grid))
            lines.append(f'  {axis}: [{values.min():.4g}, {values.max():.4g}], '
                         f'{values.size} points')
        if self.truth:
            lines.append('  truth:')
            lines += [f'    {eq}' for eq in self.truth]
            if self.truth_alternatives:
                lines.append(f'  + {len(self.truth_alternatives)} equivalent form(s)')
        else:
            lines.append('  truth: unknown')
        if self.notes:
            lines.append(f'  notes: {self.notes}')
        return '\n'.join(lines)


def mesh(*axes: Sequence[float]) -> Tuple[np.ndarray, ...]:
    """``np.meshgrid(*axes, indexing='ij')`` as a tuple, float64."""
    arrays = [np.asarray(a, dtype=np.float64) for a in axes]
    if len(arrays) == 1:
        return (arrays[0],)
    return tuple(np.meshgrid(*arrays, indexing='ij'))
