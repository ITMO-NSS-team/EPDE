"""Controlled second-axis alternatives; fitting/discrepancy stay frozen."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'outer_selection_2026-11-16'))
import run_outer as S
R,F,M,np=S.R,S.F,S.M,S.np
SV=R.SV
from dataclasses import replace


def common_width(width):
    config=R.active_config();R.set_active_config(replace(config,search_space=replace(config.search_space,
        equation_terms_max_number=width,system_rps_mode='repair')))
    assert all(SV._reference_width(p)==width for p in range(width+1))


def prediction_dyad(x,y,shape):
    if x.shape[1]==0:return 0.
    weight=np.ones(len(y));den=float(np.linalg.norm(y))
    if den==0:return 0.
    axes=[]
    for _,order,moved in SV._axis_layouts(len(y),shape):
        xx,yy=(x,y) if order is None else (x[order],y[order])
        parts=SV._dyadic_slab_edges(len(y),moved,SV._block_rows(SV._reference_width(x.shape[1])))
        fine=parts[-1];gg,gy=SV._level_gram_blocks(xx,yy,weight,moved,fine,append_constant=False)
        grams=[];rhs=[]
        for edges in parts:
            pos=np.searchsorted(fine,edges[:-1]);grams.append(np.add.reduceat(gg,pos,axis=0));rhs.append(np.add.reduceat(gy,pos,axis=0))
        design=None
        if SV._DYAD_SOLVE in SV._DESIGN_SOLVES:
            factors=SV._SlabFactors(lambda:np.column_stack([xx,yy]),weight,parts)
            def design(rows):
                rr=factors(rows);return rr[:,:,:x.shape[1]],rr[:,:,x.shape[1]]
        coef=SV._solve_min_norm(np.concatenate(grams),np.concatenate(rhs),design);offset=0;levels=[]
        for edges in parts:
            cc=coef[offset:offset+len(edges)-1];offset+=len(edges)-1
            delta=cc-np.median(cc,axis=0)
            levels.append(float(np.median(np.linalg.norm(x@delta.T,axis=0)/den)))
        axes.append(float(np.mean(levels)))
    return float(np.mean(axes))


def evaluate(matrix,candidates,shape):
    names=('dyad_sum','dyad_mean','dyad_prediction','tile','survival','chi2','complexity','discrepancy_only')
    values={name:[] for name in names};details=[];w=np.ones(len(matrix))
    for c in candidates:
        x,y=matrix[:,c['xi']],matrix[:,c['yi']]
        if len(c['xi']):
            per=SV.dyad_scores(x,y,w,shape,fit_intercept=False)
            scores=dict(dyad_sum=float(np.sum(per)),dyad_mean=float(np.mean(per)),dyad_prediction=prediction_dyad(x,y,shape))
            for name,fun in [('tile',SV.tile_scores),('survival',SV.survival_scores),('chi2',SV.chi2_scores)]:
                scores[name]=float(np.sum(fun(x,y,w,shape,fit_intercept=False)))
        else:per=np.array([]);scores={k:0. for k in names[:6]}
        scores.update(complexity=len(c['xi']),discrepancy_only=0.)
        for name,v in scores.items():values[name].append(v)
        details.append(dict(candidate=c['id'],features=len(c['xi']),dyad_per_term=per))
    assert all(np.isfinite(v).all() for v in values.values())
    return values,details


def front(d,i,cs,slot):
    ids=[j for j,c in enumerate(cs) if c['slot']==slot]
    return [j for j in ids if not any(k!=j and d[k]<=d[j] and i[k]<=i[j] and (d[k]<d[j] or i[k]<i[j]) for k in ids)]


def checks():
    common_width(3);t=np.linspace(-1,1,240);x=np.column_stack([np.ones(len(t)),t,t*t]);y=1.3-.6*t+.2*t*t
    score=prediction_dyad(x,y,(len(t),));assert score<1e-9
    drifting=y+.3*t**3;before=prediction_dyad(x,drifting,(len(t),));assert before>score
    units=np.array([.03,25.,.7]);after=prediction_dyad(x*units,drifting*4,(len(t),))
    np.testing.assert_allclose(before,after,rtol=2e-10,atol=2e-11)
    a,b=np.meshgrid(np.linspace(-1,1,30),np.linspace(-1,1,25),indexing='ij')
    xx=np.column_stack([np.ones(a.size),a.ravel()]);yy=(1.2+a+.2*b).ravel()
    original=prediction_dyad(xx,yy,a.shape);order=np.arange(a.size).reshape(a.shape).T.ravel()
    transposed=prediction_dyad(xx[order],yy[order],a.T.shape)
    np.testing.assert_allclose(original,transposed,rtol=2e-10,atol=2e-11)
    assert original>.001
    R.dump(ROOT/'checks.json',dict(exact_constant_law_stable=True,feature_target_units_invariant=True,
        axis_transposition_invariant=True,spatial_drift_detected=True,drift_score=before,source_sha256=S.sha(__file__)))
    print('Prediction dyad: constants, units, drift, axis symmetry PASS')


if __name__=='__main__':checks()
