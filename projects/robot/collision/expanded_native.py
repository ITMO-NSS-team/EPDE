"""NSGA-II with one physical contact vector and up to two scalar robot modifiers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
import random
import subprocess
import sys
from unittest.mock import patch

from expanded_data import CACHE, DEST, PROJECTED, fingerprint
from physics import OUT
from expanded_terms import GATES, IDS, NATIVE_VECTORS
from discovery import write_json
import numpy as np

RHS = 10
FACTORS = 3
EPOCHS = 12
POPULATION = 16
SEED = 923701


def read_screen(held):
    sample = []
    for path in sorted(CACHE.glob(f'{held}__*__train.npz')):
        with np.load(path) as z:
            assert str(z['digest']) == fingerprint()
            sample.append(z['sample'])
    assert len(sample)==4
    return np.concatenate(sample)


def label(term,allowed_vectors,gate_names=GATES,modifiable=NATIVE_VECTORS):
    labels = [f.label for f in term.structure]
    vectors = [v for v in labels if v in allowed_vectors]
    gates = [g for g in gate_names if g in labels]
    if len(vectors)!=1 or len(gates)+1!=len(labels) or any(f.params[0]!=1 for f in term.structure):
        return None
    if gates and vectors[0] not in modifiable:
        return None
    return '*'.join([*vectors,*gates])


def run(held, library, target_check_mode='direct'):
    a = read_screen(held)
    y = a[:,len(IDS)]
    vectors = NATIVE_VECTORS if library=='focused' else [v for v in IDS if '*' not in v]
    x = {v:a[:,IDS.index(v)] for v in vectors}
    direct = library=='broad' and target_check_mode=='direct'
    receipt = search_screen(y,x,{g:a[:,len(IDS)+1+k] for k,g in enumerate(GATES)},NATIVE_VECTORS,
                            lambda name:a[:,IDS.index(name)],direct=direct)
    prefix = '' if library=='focused' else 'broad_'
    write_json(DEST/f'{held}_{prefix}nsga2.json',dict(held=held,digest=fingerprint(),library=library,
               screen_sha256=hashlib.sha256(a.tobytes()).hexdigest(),**receipt))


def search_screen(y,x,gates,modifiable,column,direct=True):
    from epde.interface.interface import EpdeSearch
    from epde.interface.prepared_tokens import CacheStoredTokens
    np.random.seed(SEED)
    random.seed(SEED)
    ep = EpdeSearch(use_solver=False,multiobjective_mode=True,engine='nsga2',system_rps_mode='plain',
                    discrepancy_metric='wape',instability_metric='dyad',device='cpu',
                    system_rps_params=dict(variable_order_seed=SEED,target_axes=0,fill_probability=0.),
                    verbose_params=dict(show_iter_idx=False,candidate_objectives=False))
    ep.set_moeadd_params(population_size=POPULATION,training_epochs=EPOCHS,early_stopping_patience=None)
    _, dom = ep.createDomain((np.arange(len(y),dtype=float),),boundary_width=0,ID=0)
    _, tr = ep.createTrajectory({'u':np.zeros_like(y)},dom,cache_id=0,derivs=y[:,None])
    args = dict(params_ranges={'power':(1,1)})
    tokens = [CacheStoredTokens('vector',list(x),token_tensors=x,meaningful=True,**args)]
    for g,values in gates.items():
        tokens.append(CacheStoredTokens(g+'Family',[g],token_tensors={g:values},**args))
    with physical_candidates(y,x,gates,modifiable,column,direct) as (archive,item):
        ep.fit(data=[tr],max_deriv_order=(1,),data_fun_pow=1,deriv_fun_pow=1,
               equation_terms_max_number=RHS+1,equation_factors_max_number=FACTORS,additional_tokens=tokens)
    front = [item(s.vals['u']) for s in ep.equations(only_print=False)[0] if np.isfinite(s.obj_fun).all()
             and s.vals['u'].weights_final_evald and s.obj_fun[0]<1e5]
    assert len(ep.optimizer._pareto_history)==EPOCHS and front and archive
    return dict(seed=SEED,target_check='direct' if direct else 'enumerated',
               epochs=EPOCHS,population=int(ep.optimizer._mu),rhs_terms=RHS,max_factors=FACTORS,
               front=front, evaluated=list(archive.values()))


@contextmanager
def physical_candidates(y,x,gates,modifiable,column,direct=True):
    """Apply the study's RHS grammar for one fit and collect the structures EPDE evaluates."""
    from epde.operators.common import invariant_terms, plain_fit

    archive = {}

    def physical_only(equation, features):
        return [label(term,x,gates,modifiable) is None for term in features]

    def item(eq):
        terms = [label(t,x,gates,modifiable) for k,t in enumerate(eq.structure) if k!=eq.target_idx]
        beta = dict((k,float(v)) for k,v in zip(terms,eq.weights_final[:-1]) if v!=0)
        assert None not in beta and len(beta)>0 and eq.weights_final[-1]==0
        pred = sum(column(k).astype(float)*v for k,v in beta.items())
        wape = np.abs(y-pred).sum()/np.abs(y.astype(float)).sum()
        np.testing.assert_allclose(wape, eq.fitness_value,rtol=1e-5,atol=1e-6)
        return dict(terms=beta,wape=float(eq.fitness_value),dyad=float(eq.coefficients_stability))

    def observe(*args,**kwargs):
        versions = plain_fit._linear_versions(*args,**kwargs,intercept=False)
        for eq in versions:
            solution = item(eq)
            archive[tuple(sorted(solution['terms']))] = solution
        return versions

    def check_forecast_target(eq,axes=None):
        # There is exactly one supplied derivative. Verify its family, power and
        # axis directly instead of walking all possible three-factor RHS terms
        # each time EPDE copies the pool. The target remains EPDE's plain derivative.
        for fam in eq.pool.families_demand_equation:
            if (fam.variable==eq.main_var_to_explain and set(fam.token_params)=={'power'} and
                    fam.token_params['power'][0]<=1<=fam.token_params['power'][1] and
                    (axes is None or 0 in axes) and
                    any(list(code)==[0] for code in fam.derivs_ords.values())):
                return
        raise ValueError('Forecast derivative absent from the EPDE token pool')

    # Use the physical RHS grammar before EPDE fits each candidate. In
    # particular a product involving the derivative target can never be a predictor.
    target_check = patch.object(plain_fit,'_check_plain_tokens',check_forecast_target) if direct else nullcontext()
    with patch.object(invariant_terms,'refused',physical_only),patch.object(plain_fit,'_versions',observe),target_check:
        yield archive,item


def launch(held,library):
    prefix = '' if library=='focused' else 'broad_'
    log = DEST/f'{held}_{prefix}nsga2.log'
    path = log.with_suffix('.json')
    # The target check enumerates tokens in older receipts and consumes
    # random numbers for their internal hashes. Replay the recorded choice.
    previous = json.loads(path.read_text()) if path.exists() else {}
    if PROJECTED and previous and all(previous.get(k)==v for k,v in
            dict(digest=fingerprint(),seed=SEED,library=library,epochs=EPOCHS,population=POPULATION,
                 rhs_terms=RHS,max_factors=FACTORS,held=held).items()):
        return
    reference = OUT/'expanded_factors'/path.name
    replay = json.loads(reference.read_text()) if PROJECTED and reference.exists() else previous
    mode = replay.get('target_check','direct')
    with log.open('w',encoding='utf-8') as f:
        proc = subprocess.run([sys.executable,'-u',__file__,'--fold',held,'--library',library,'--target-check',mode],
                              stdout=f,stderr=subprocess.STDOUT,
                              env=dict(os.environ,PYTHONHASHSEED='0'))
    if proc.returncode:
        raise RuntimeError(log.read_text(errors='replace')[-2200:])
    data = json.loads((DEST/f'{held}_{prefix}nsga2.json').read_text())
    print(held,len(data['evaluated']),'native structures;',len(data['front']),'front members',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold')
    p.add_argument('--library',choices=['focused','broad'],default='focused')
    p.add_argument('--target-check',choices=['direct','enumerated'],default='direct')
    args = p.parse_args()
    if args.fold:
        run(args.fold,args.library,args.target_check)
    else:
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(lambda held:launch(held,args.library),sorted(p.name.split('__')[0] for p in CACHE.glob('*__test.npz'))))
