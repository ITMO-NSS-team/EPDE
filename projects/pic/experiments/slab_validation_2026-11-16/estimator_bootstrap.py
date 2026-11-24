"""Cache data-only parametric-bootstrap fields for corrected weak regression.

No equations or clean truth are passed to the estimator in a noisy-data run.
Clean records are separate controls processed by the identical pipeline.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import time

import signal_refits as S
import noise_support as N
import window_profiles as W
import numpy as np


def bootstrap(t, observed, count, seed, system):
    pilot = S.smoothing(t, observed)
    model = W.WF.noise_model(observed)
    index = {"rational": 0, "kepler": 1, "cauchy": 2}[system]
    errors, smoothed = N.propagate(t, pilot, model, count, seed, index)
    return dict(t=t, observed=observed, pilot=pilot, errors=errors, smoothed_draws=smoothed,
                sigma=np.asarray(model[0]), rho=np.asarray(model[1]), difference_order=np.asarray(model[2]))


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = json.loads((args.input_dir / "signal_refits.json").read_text(encoding="utf-8"))
    done = set()
    for case in saved["cases"]:
        for trial in case["trials"]:
            path = args.input_dir / f"{case['run']}_seed{trial['seed']}_signal.npz"
            for control in (False, True) if trial["seed"] == 0 else (False,):
                tag = "clean" if control else f"seed{trial['seed']}"
                key = f"{case['system']}_{tag}"
                if key in done:
                    continue
                done.add(key)
                with np.load(path) as source:
                    t = source["t"].copy()
                    observed = source["clean" if control else "noisy"].copy()
                metadata = dict(system=case["system"], tag=tag, draws=args.draws,
                                observed_digest=W.digest(observed),
                                source_hashes={str(p): W.source_hash(p) for p in [Path(__file__), Path(S.__file__),
                                    Path(N.__file__), W.REPO / "epde/preprocessing/weak_form.py"]})
                output = args.output_dir / f"{key}_bootstrap.npz"
                if output.exists():
                    with np.load(output) as old:
                        if json.loads(str(old["metadata"])) != metadata:
                            raise ValueError(f"Stale cache: {output}")
                    print(f"Validated existing {key}", flush=True)
                    continue
                start = time.perf_counter()
                print(f"Generating {key}: {args.draws} draws", flush=True)
                payload = bootstrap(t, observed, args.draws, 1000000 if control else trial["seed"], case["system"])
                np.savez_compressed(output, **payload, metadata=json.dumps(metadata))
                print(f"Saved {key}: sigma={float(payload['sigma']):.6g}, rho={payload['rho'].tolist()}, "
                      f"{time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
