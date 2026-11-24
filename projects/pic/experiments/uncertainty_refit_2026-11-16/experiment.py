"""Matched pilot correction and adaptive reconstruction/coefficient refits."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'dyad_ablation_2026-11-16'))
import criteria as K
R,F,M,np,S=K.R,K.F,K.M,K.np,K.S
J,O,Q=M.J,M.O,M.Q
import json,time,argparse
B=128


class Smoother:
    def __init__(self,raw,axes,info):
        n=len(axes[0]);time_d,u,ib=R.spline_basis(tuple(axes[0]),tuple(range(n)));self.maps={};self.spatial={}
        for var,y in raw.items():
            lam=info[var][0]['lam'];f=(time_d==0).astype(float) if np.isinf(lam) else 1/(1+lam*time_d)
            self.maps[var]=(ib*f)@u.T;self.spatial[var]=[]
            for axis in range(1,y.ndim):
                d=(np.arange(y.shape[axis],dtype=float)/y.shape[axis])**4;lam=info[var][axis]['lam']
                self.spatial[var].append((d==0).astype(float) if np.isinf(lam) else 1/(1+lam*d))
    def __call__(self,values):
        result={}
        for var,y in values.items():
            z=np.tensordot(self.maps[var],y,axes=(1,0))
            for axis,f in enumerate(self.spatial[var],1):
                shape=[1]*z.ndim;shape[axis]=len(f)
                z=R.idct(R.dct(z,axis=axis,norm='ortho')*f.reshape(shape),axis=axis,norm='ortho')
            result[var]=z
        return result


class Experiment:
    def __init__(self,name,seed):
        self.name,self.seed=name,seed;self.start=time.monotonic();basepath,self.old=F.baseline(name,0)
        paths=[basepath,M.ROOT/'results'/f'{name}_seed0.json',J.ROOT/'results'/f'{name}_seed0.json',
               F.ROOT/'fit_results'/f'{name}_seed0.json',K.ROOT/'results'/f'{name}_seed0.json']
        self.inputs={str(p):S.sha(p) for p in paths};self.prior,self.joint,self.local,self.ablation=[json.loads(p.read_text()) for p in paths[1:]]
        self.p=R.datasets.load(name);self.axes=R.axes_of(self.p);self.cs,self.labels=R.panel(self.p)
        self.terms,self.opaque=R.symbols(self.p,self.labels);K.common_width(max(len(c['xi']) for c in self.cs))
        self.raw=self.p.noisy(5,seed);self.indices=[np.asarray(g['indices']) for g in self.old['groups']]
        assert all(not np.intersect1d(a,b).size for i,a in enumerate(self.indices) for b in self.indices[:i])
        self.vseg=slice(*self.prior['segment']);ax=[self.axes[0][self.vseg],*self.axes[1:]]
        opq={k:y[self.vseg] for k,y in self.opaque.items()};self.keys=[f['selection']['q0']['key'] for f in self.prior['folds']]
        self.vbanks={};self.basis={}
        for key in set(self.keys):
            meta=self.prior['banks'][key];self.vbanks[key]=S.N.V.make_broad(ax,self.terms,opq,meta['plan'],meta['k'])
            self.basis[key]=M.RealBasis(meta['k'],len(ax))
        self.states=[];self.choices=[];self.anchor_vectors=[];self.replay_coefficient_error=0.
        for g,idx in enumerate(self.indices):
            obs={v:y[idx] for v,y in self.raw.items()};model={v:R.WF.noise_model(y) for v,y in obs.items()}
            pilot,info=R.reconstruct(obs,self.axes,idx);tm=O.interpolation(self.axes[0],idx)[self.vseg];cov=O.Covariance(ax,tm,model)
            seg,tax,topq,plan=Q.mapped_plan(self.old,g,self.local['plans'][g]['plan'],self.axes,self.opaque)
            bankset={'local':Q.Bank(tax,self.terms,topq,plan=plan)}
            for tag,meta in self.joint['folds'][g]['banks'].items():
                if meta['feasible']:bankset[tag]=S.N.V.make_broad(tax,self.terms,topq,meta['plan'],meta['k'])
            choices={r['candidate']:r['selected']['all'] for r in self.joint['folds'][g]['choices'] if r['draws']==128}
            self.choices.append([choices[c['id']] for c in self.cs])
            needed={v.rsplit('_',1)[0] for v in choices.values()};bankset={k:v for k,v in bankset.items() if k in needed}
            matrices={tag:[bank.matrix({v:y[seg] for v,y in pilot.items()})] for tag,bank in bankset.items()}
            rng=np.random.default_rng(np.random.SeedSequence([1171,seed,g]))
            for b in range(self.old['draws']):
                draw={v:pilot[v][idx]+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in obs.items()}
                fitfield,_=R.reconstruct(draw,self.axes,idx)
                for tag,bank in bankset.items():matrices[tag].append(bank.matrix({v:y[seg] for v,y in fitfield.items()}))
            prepared={tag:J.prepare(np.stack(mm),self.old['draws']) for tag,mm in matrices.items()}
            corrections={tag:dict(bias=st['p']-st['z'],gram=st['correction']) for tag,st in prepared.items()}
            vectors=np.zeros((len(self.labels),len(self.cs)))
            for j,c in enumerate(self.cs):
                tag,kind=self.choices[g][j].rsplit('_',1);st=prepared[tag]
                coef,_=R.signed_fit(st['point_'+kind],st['p'],c['xi'],c['yi'])
                vectors[c['yi'],j]=1.;vectors[c['xi'],j]-=coef
                if seed==0:
                    expected=self.joint['folds'][g]['options'][j]['options'][self.choices[g][j]]['coefficients']
                    np.testing.assert_allclose(coef,expected,rtol=5e-7,atol=2e-8)
                    if len(coef):self.replay_coefficient_error=max(self.replay_coefficient_error,float(np.max(np.abs(coef-expected))))
            self.anchor_vectors.append(vectors)
            own=self.keys[g];vbank=self.vbanks[own];basis=self.basis[own]
            vp=basis.convert(vbank.matrix({v:y[self.vseg] for v,y in pilot.items()}));draws=[]
            rng=np.random.default_rng(np.random.SeedSequence([834721,seed,g]))
            for b in range(B):
                draw={v:pilot[v][idx]+R.WF.ar_noise(rng,y.shape,*model[v][:2]) for v,y in obs.items()}
                draws.append(basis.convert(O.matrix(vbank,O.map_values(tm,draw),cov)))
            white,wmeta=M.whitening(np.stack(draws),vp);assert white is not None
            self.states.append(dict(map=tm,cov=cov,white=white,scale=np.sum((white@vp)**2,axis=0)[[c['yi'] for c in self.cs]],
                banks=bankset,segment=seg,corrections=corrections,noise=model,reconstruction=info,weight_meta=wmeta))
        self.anchor_vectors=np.stack(self.anchor_vectors)
        plan={k:tuple(v) for k,v in self.old['base_plan'].items()};self.dbank=R.LocalBank(self.axes,self.terms,self.opaque,plan);self.dshape=plan['shape']
        self.pilot,self.pilotinfo=R.reconstruct(self.raw,self.axes,np.arange(len(self.axes[0])))
        self.smoother=Smoother(self.raw,self.axes,self.pilotinfo);smooth=self.smoother(self.raw)
        for v in smooth:np.testing.assert_allclose(smooth[v],self.pilot[v],rtol=2e-10,atol=2e-11)
        constants=self.smoother({v:np.ones_like(y) for v,y in self.raw.items()})
        for y in constants.values():np.testing.assert_allclose(y,1.,rtol=2e-10,atol=2e-11)
        twice_smoothed=self.smoother(self.pilot);self.twice={v:2*y-twice_smoothed[v] for v,y in self.pilot.items()}
        self.model={v:R.WF.noise_model(y) for v,y in self.raw.items()}
        self.clean_grams=None

    def evaluate(self,fields,correct=True):
        matrix=self.dbank.matrix(fields);dyad=[];weight=np.ones(len(matrix))
        for c in self.cs:
            with np.errstate(over='ignore',invalid='ignore'):
                dyad.append(float(np.sum(R.SV.dyad_scores(matrix[:,c['xi']],matrix[:,c['yi']],weight,self.dshape,fit_intercept=False))) if c['xi'] else 0.)
        points=[];refitted=[];negative=0;deficient=0
        for g,(idx,st) in enumerate(zip(self.indices,self.states)):
            obs={v:y[idx] for v,y in fields.items()};mapped=O.map_values(st['map'],obs)
            points.append({k:self.basis[k].convert(O.matrix(bank,mapped,st['cov']) if correct else bank.matrix(mapped)) for k,bank in self.vbanks.items()})
            pilot,_=R.reconstruct(obs,self.axes,idx);seg=st['segment'];pp={v:y[seg] for v,y in pilot.items()}
            mm={tag:bank.matrix(pp) for tag,bank in st['banks'].items()};vectors=np.zeros_like(self.anchor_vectors[g])
            for j,c in enumerate(self.cs):
                tag,kind=self.choices[g][j].rsplit('_',1);m=mm[tag];corr=st['corrections'][tag]
                if kind=='clsm' and correct:
                    z=m-corr['bias'];gram=J.grams(z)-corr['gram']
                else:gram=J.grams(m)
                coef,info=R.signed_fit(gram,m,c['xi'],c['yi']);negative+=info['negative']>0;deficient+=info['rank']<len(c['xi'])
                vectors[c['yi'],j]=1.;vectors[c['xi'],j]-=coef
            refitted.append(vectors)
        refitted=np.stack(refitted);scores=[]
        for vectors in (self.anchor_vectors,refitted):
            folds=[]
            for g,st in enumerate(self.states):
                left,right=[h for h in range(3) if h!=g];key=self.keys[g]
                a=st['white']@points[left][key]@vectors[g];b=st['white']@points[right][key]@vectors[g]
                folds.append(np.sum(a*b,axis=0)/st['scale'])
            scores.append(np.mean(folds,axis=0))
        scores=np.stack(scores);dyad=np.array(dyad)
        assert np.isfinite(scores).all() and np.isfinite(dyad).all() and np.isfinite(refitted).all()
        return dict(discrepancy=scores,dyad=dyad,vectors=refitted,negative=negative,deficient=deficient)

    def energy(self,vectors):
        energies=[]
        for g,v in enumerate(vectors):
            gram=self.clean_grams[f'{g}_loo_oracle'];scale=self.clean_grams[f'{g}_loo_clean_scale'][[c['yi'] for c in self.cs]]
            energies.append(np.einsum('ij,ik,kj->j',v,gram,v)/scale)
        return np.mean(energies,axis=0)


def sources():
    paths=[Path(__file__),ROOT/'protocol.txt',Path(K.__file__),Path(R.__file__),Path(R.WF.__file__),Path(R.SV.__file__),
        Path(J.__file__),Path(Q.__file__),Path(O.__file__),Path(M.__file__),Path(F.__file__)]
    return {str(p):S.sha(p) for p in paths}


def run(name,seed,replay=False):
    e=Experiment(name,seed);observed=e.evaluate(e.raw)
    np.testing.assert_allclose(observed['vectors'],e.anchor_vectors,rtol=5e-8,atol=2e-8)
    np.testing.assert_allclose(observed['discrepancy'][0],observed['discrepancy'][1],rtol=1e-8,atol=2e-8)
    if replay:
        assert seed==0
        np.testing.assert_allclose(observed['discrepancy'][0],e.ablation['discrepancy']['q0_white128/joint'],rtol=5e-7,atol=2e-8)
        np.testing.assert_allclose(observed['dyad'],e.ablation['instability']['dyad_sum'],rtol=2e-10,atol=2e-11)
        R.dump(ROOT/'replay'/f'{name}.json',dict(status='PASS',system=name,coefficient_error=e.replay_coefficient_error,
            discrepancy_error=float(np.max(np.abs(observed['discrepancy'][0]-e.ablation['discrepancy']['q0_white128/joint']))),
            source_sha256=sources(),input_sha256=e.inputs,seconds=time.monotonic()-e.start))
        print(name,'REPLAY PASS',round(time.monotonic()-e.start,1),'s',flush=True);return
    # All design choices, fits, pilots and uncertainty models are now frozen.
    frozen=dict(system=name,seed=seed,candidates=e.cs,choices=e.choices,anchor_vectors=e.anchor_vectors,
        model=e.model,pilot_info=e.pilotinfo,states=[{k:st[k] for k in ('noise','reconstruction','weight_meta')} for st in e.states],
        raw_data_sha256={v:R.digest(y) for v,y in e.raw.items()},
        pilot_sha256={tag:{v:R.digest(y) for v,y in fields.items()} for tag,fields in [('standard',e.pilot),('twice',e.twice)]},
        input_sha256=e.inputs,source_sha256=sources())
    freeze_path=ROOT/'frozen'/f'{name}_seed{seed}.json';R.dump(freeze_path,frozen);freeze_hash=S.sha(freeze_path)
    e.clean_grams=F.evaluation_grams(e.p,e.old,e.terms,e.opaque,e.labels)
    centers={tag:e.evaluate(fields,correct=False) for tag,fields in [('standard',e.pilot),('twice',e.twice),('clean',e.p.data)]}
    samples={k:[] for k in ('standard','twice','benchmark')};rng=np.random.default_rng(np.random.SeedSequence([460713,seed]));innovation_hashes=[]
    for b in range(B):
        innovation={v:R.WF.ar_noise(rng,y.shape,*e.model[v][:2]) for v,y in e.raw.items()}
        if b in (0,B-1):innovation_hashes.append({v:R.digest(y) for v,y in innovation.items()})
        for tag,pilot in [('standard',e.pilot),('twice',e.twice)]:
            sample=e.evaluate({v:y+innovation[v] for v,y in pilot.items()});sample['energy']=e.energy(sample['vectors']);samples[tag].append(sample)
        sample=e.evaluate(e.p.noisy(5,620000+1000*seed+b));sample['energy']=e.energy(sample['vectors']);samples['benchmark'].append(sample)
        if (b+1)%32==0:print(name,seed,'triples',b+1,round(time.monotonic()-e.start,1),'s',flush=True)
    arrays={'anchor_vectors':e.anchor_vectors,'anchor_energy':e.energy(e.anchor_vectors)}
    for label,st in [('observed',observed),*[(k+'_center',v) for k,v in centers.items()]]:
        for key,value in st.items():arrays[label+'_'+key]=value
    for label,ss in samples.items():
        for key in ss[0]:arrays[label+'_'+key]=np.stack([s[key] for s in ss])
    path=ROOT/'results'/f'{name}_seed{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    assert S.sha(freeze_path)==freeze_hash
    R.dump(path.with_suffix('.json'),dict(system=name,seed=seed,draws=B,candidates=e.cs,
        scope='Frozen geometry/method/weight/correction choices; main refits include adaptive GCV reconstruction.',
        frozen_sha256=freeze_hash,array_sha256=S.sha(path),source_sha256=sources(),input_sha256=e.inputs,
        innovation_hashes=innovation_hashes,actual_noise_seeds=[620000+1000*seed+b for b in range(B)],
        pilot_relative_l2_error={tag:{v:float(np.linalg.norm(fields[v]-y)/np.linalg.norm(y)) for v,y in e.p.data.items()}
            for tag,fields in [('standard',e.pilot),('twice',e.twice)]},
        seconds=time.monotonic()-e.start))
    print(name,seed,'DONE',round(time.monotonic()-e.start,1),'s',flush=True)


if __name__=='__main__':
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--system',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    out=ROOT/('replay' if a.replay else 'results')/(f'{a.system}.json' if a.replay else f'{a.system}_seed{a.seed}.json')
    if a.replay or not out.exists():run(a.system,a.seed,a.replay)
