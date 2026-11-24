"""Cross fitted coefficients and held-out donors on disjoint seed halves."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
AC=ROOT.parent/"ac_diffusion_uncertainty_2026-11-18"


def table(terms,observations,scales):
    numerator=np.einsum("ifka,jfab,ifkb->ijfk",terms,observations,terms,optimize=True)
    denom=np.einsum("ifka,jfa->ijfk",terms**2,scales**2,optimize=True)
    scores=(numerator/denom).mean(axis=2)
    return scores[:,:,1]-scores[:,:,0]


def summarize(matrix):
    grand=np.mean(matrix)
    row=np.mean(matrix,axis=1)-grand
    col=np.mean(matrix,axis=0)-grand
    residual=matrix-grand-row[:,None]-col[None,:]
    components=np.array([np.mean(row**2),np.mean(col**2),np.mean(residual**2)])
    np.testing.assert_allclose(sum(components),np.var(matrix),rtol=1.e-12,atol=1.e-24)
    parts=components/np.var(matrix)
    return dict(n_coefficients=len(row),n_heldouts=len(col),mean=float(grand),positive=float(np.mean(matrix>0)),
        variance=float(np.var(matrix)),coefficient_share=parts[0],heldout_share=parts[1],interaction_share=parts[2])


def run(directory):
    pieces=[]
    for path in sorted((directory/"draws").glob("fresh_*.npz")):
        with np.load(path) as z:
            pieces.append({k:z[k] for k in ("seeds","vec","cross","scale","scores")})
    data={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    assert len(data["seeds"])==128 and np.all(np.diff(data["seeds"])==1)
    margins=table(data["vec"][:,:,:2],data["cross"],data["scale"])
    observed=data["scores"].mean(axis=1)
    err=float(np.max(abs(np.diag(margins)-(observed[:,1]-observed[:,0]))))
    assert err<1.e-11
    return dict(even_coefficients=summarize(margins[::2,1::2]),odd_coefficients=summarize(margins[1::2,::2]),
                maximum_diagonal_error=err)


if __name__=="__main__":
    out={name:run(d) for name,d in (("allen_cahn",AC),("navier_stokes",ROOT))}
    (ROOT/"crossed.json").write_text(json.dumps(out,indent=2))
    print(out)
