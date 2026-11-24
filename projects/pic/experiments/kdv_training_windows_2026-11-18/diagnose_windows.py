"""Compare noisy training dispersion moments with their clean benchmark values."""
import json
from holdouts import ROOT, E, np
from training import geometry
from compare import source


def clean_blocks(system):
    problem, _, _, common, folds, _ = geometry(system)
    out = {}
    for g, (_, seg, local, time, spatial, direction) in enumerate(folds):
        part = {k: v[seg] for k, v in problem.data.items()}
        spatial_matrix = spatial.matrix(part)
        blocks = dict(local=local.matrix(part), time=time.matrix({k: v[common] for k, v in problem.data.items()}),
                      spatial=spatial_matrix)
        if direction is not None:
            blocks["forcing"] = np.einsum("ij,ijk->ik", direction.conj(), spatial_matrix.reshape(*spatial.shape, -1))
        for tag, value in blocks.items():
            out[f"{g}_{tag}_point"] = value
    return out


if __name__ == "__main__":
    clean = clean_blocks("kdv_cossin")
    by_design = {name: [] for name in ("original", "wide", "projected")}
    shares = {name: [] for name in by_design}
    _, ulabels = E.R.panel(E.R.datasets.load("kdv"))
    for dest, first in ((ROOT, 788001), (ROOT / "projection_check", 790001)):
        fmeta = json.loads((dest / "records" / f"kdv_cossin_{first}.json").read_text())
        ref = next(c for c in fmeta["candidates"] if c["label"] == "reference")
        col = ref["xi"][1]
        assert "d^3u/dx1^3" in str(fmeta["labels"][col])
        for i in range(24):
            forced, _ = source(dest / "training", f"kdv_cossin_{first+i}")
            unforced, uinfo = source(dest / "training", f"kdv_{first+1000+i}")
            j, = [j for j, k in enumerate(uinfo["columns"]) if "d^3u/dx1^3" in str(ulabels[k])]
            for name, tags in (("original", ("local","time")), ("wide", ("spatial","spatial")),
                               ("projected", ("spatial","forcing"))):
                field_ratios, field_shares = [], []
                for g in range(3):
                    def vector(data, tags, column):
                        a = [data[f"{g}_{t}_point"][:,column] for t in tags]
                        return np.concatenate([x / np.sqrt(len(x)) for x in a])
                    pilot = vector(forced, tags, col)
                    exact = vector(clean, tags, col)
                    intrinsic = vector(unforced, ("local", "time"), j)
                    ue = np.vdot(intrinsic, intrinsic).real
                    field_shares.append(ue / (ue + np.vdot(pilot, pilot).real))
                    field_ratios.append(np.linalg.norm(pilot-exact)/np.linalg.norm(exact))
                by_design[name].append(np.median(field_ratios))
                shares[name].append(np.median(field_shares))
    out = {name: dict(fields=len(values), median_relative_dispersion_moment_error=float(np.median(values)),
                      median_unforced_dispersion_norm_fraction=float(np.median(shares[name]))) for name, values in by_design.items()}
    (ROOT / "dispersion_diagnostic.json").write_text(json.dumps(out,indent=2))
    print(out)
