"""Check every loader; return nonzero for invalid data, grids or truths."""
import sys
import numpy as np
from epde_bench import datasets, metrics
from epde_bench.preprocessing import validate_data


def main():
    failures = 0
    for name in datasets.names('all'):
        try:
            p = datasets.load(name)
            validate_data(p, p.data)
            canon = [metrics.canonical_tokens(s) for s in p.truth_systems]
            if canon and len(canon[0]) != len(p.truth):
                raise ValueError('truth equations lost in parsing')
            if any(a.shape != p.shape or not np.isfinite(a).all()
                   for a, _ in p.named_arrays.values()):
                raise ValueError('extra arrays must have the data shape and finite values')
            print(f'OK   {name:16} shape={p.shape} truth_terms='
                  f'{[sum(len(eq) for eq in c) for c in canon]}')
        except FileNotFoundError as exc:
            if name == 'darcy':
                print(f'SKIP {name:16} expected missing data: {exc}')
            else:
                failures += 1
                print(f'FAIL {name:16} {exc}')
        except Exception as exc:
            failures += 1
            print(f'FAIL {name:16} {type(exc).__name__}: {exc}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
