"""Compare catalog trajectories with the stored clean benchmarks at their original start."""
import json
from settings import DATASETS, ROOT, limit_threads

if __name__ == '__main__':
    limit_threads()

from validate_trajectories import dynamics, problems
from noise_suite import atomic_text, live
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton


if __name__ == '__main__':
    output = {}
    for name in DATASETS:
        original, _ = problems(name)
        t, = live.R.axes_of(original)
        observed = np.vstack([original.data[k] for k in original.variables])
        initial = observed[:, 0].tolist()
        rhs = dynamics(name, original)
        if len(initial) == 1:
            def mismatch(velocity):
                sol = solve_ivp(rhs, t[:2], [initial[0], velocity], method='DOP853', rtol=1e-12, atol=1e-13)
                return sol.y[0,-1]-observed[0,1]
            initial.append(float(newton(mismatch, (observed[0,1]-initial[0])/(t[1]-t[0]))))
        sol = solve_ivp(rhs, (t[0], t[-1]), initial, t_eval=t, method='DOP853', rtol=1e-11, atol=1e-13)
        assert sol.success
        actual = observed[:,2:]
        residual = sol.y[:len(observed),2:] - actual
        relative_error = np.linalg.norm(residual, axis=1) / np.linalg.norm(actual-actual.mean(axis=1)[:,None], axis=1)
        output[name] = dict(initial=initial, relative_error=relative_error.tolist())
    atomic_text(ROOT / 'simulator_check.json', json.dumps(output, indent=2, allow_nan=False))
