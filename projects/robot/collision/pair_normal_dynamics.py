"""Fold-local propulsion and prior vector forecasts for the normal-velocity correction law."""
import json

from pair_normal_data import PREVIOUS, STUDY
from expanded_terms import IDS, WARMUP, columns, dot, history, pos
from physics import CAP, DT, LAG, arena_for, fit_self, heading, robot_parameters, rotate
from projected_data import fold_calibrations, projected_tracks
import numpy as np
import pandas as pd


DIR = STUDY/'dynamics'
NAMES = ('prior','priorOpening','priorClosing','capCorrection','capCurvedCorrection','priorBend',
         'freeNormal','freeOpening','freeClosing','driveNormal','driveOpening','driveClosing',
         'driveCapCorrection','driveBend','freeCapCorrection','individualIn','turnedDrive','turnedDriveCap')


def normal_dynamics(clean,rr,f,i,j,n,calibration,base,pred):
    at = rr[f]
    t = rotate(n)
    r = np.linalg.norm(clean.pos[at,i]-clean.pos[at,j],axis=1)
    gap = pos(r-CAP)/DT
    ph = clean.phi[rr]
    def drive(angle):
        e = heading(angle)
        return calibration[:,0,None]*e+calibration[:,1,None]*rotate(e)
    p = drive(ph)
    omega = (ph-clean.phi[rr-LAG])/DT
    rotated = drive(ph+(omega-calibration[:,2])*DT/2)
    def project(v,axis):
        return dot(v[f,i]-v[f,j],axis)
    u,ut = project(pred,n),project(pred,t)
    un,us = project(base,n),project(base,t)
    dn,ds = project(p,n),project(p,t)
    tn,ts = project(rotated,n),project(rotated,t)
    def bend(s):
        return s*s*DT/(2*r)
    return np.column_stack((u,pos(u),pos(-u),pos(-gap-u),pos(-gap-bend(ut)-u),bend(ut),
            un,pos(un),pos(-un),dn,pos(dn),pos(-dn),pos(-gap-bend(ds)-dn),bend(ds),
            pos(-gap-bend(us)-un),pos(-dot(p[f,i],n))+pos(dot(p[f,j],n)),tn,pos(-gap-bend(ts)-tn)))


def prepare(pooled_only=False):
    raw,tracks = projected_tracks()
    pars = fold_calibrations(tracks)
    terms = json.loads((PREVIOUS/'summary.json').read_text())['minimum_wape_terms']
    indices = [IDS.index(t) for t in terms]
    weights = pd.read_csv(PREVIOUS/'cross_recording'/'projected_by_recording.csv.gz',usecols=['terms','held','refit'])
    weights = weights[weights.terms=='+'.join(terms)].set_index('held').refit
    prefixes = [tr.part(0,3*len(tr.pos)//4) for tr in tracks.values()]
    arenas = arena_for(prefixes)
    pars['pooled'] = (*fit_self(prefixes,arenas),arenas)
    weights['pooled'] = json.dumps(json.loads((PREVIOUS/'cross_recording'/'pooled_common_law.json').read_text())['terms'])
    DIR.mkdir(exist_ok=True)
    for tag,tr in tracks.items():
        cut = len(tr.pos)*3//4
        for role,lo,hi in [('train',0,cut),('valid',cut,None),('test',0,None)]:
            if pooled_only and role!='train':
                continue
            clean,obs = tr.part(lo,hi),raw[tag].part(lo,hi)
            rows = np.arange(WARMUP,len(clean.pos)-LAG,LAG)
            with np.load(STUDY/f'{tag}__{role}.npz') as d:
                origins = d['origins']
                previous = d['previous'] if role=='test' else None
            ids = {rid:k for k,rid in enumerate(clean.ids)}
            a = origins[:,0]-lo
            first = np.array([ids[k] for k in origins[:,1]])
            second = np.array([ids[k] for k in origins[:,2]])
            memory = history(clean,rows,obs)
            holds = [tag] if role=='test' else sorted(set(tracks)-{tag})
            if role=='train':
                holds.append('pooled')
            for held in (['pooled'] if pooled_only else holds):
                b = [json.loads(weights[held])[t] for t in terms]
                calibration = robot_parameters(clean,pars[held][0])
                blocks = []
                for start in range(0,len(rows),48):
                    sl = np.arange(start,min(start+48,len(rows)))
                    rr = rows[sl]
                    keep = slice(*np.searchsorted(a,[rr[0],rr[-1]+1]))
                    at,i,j = a[keep],first[keep],second[keep]
                    f = np.searchsorted(rr,at)
                    xx,base,*_ = columns(clean,rr,sl,memory,pars[held],obs)
                    pred = base+xx[...,indices]@b
                    raw_d = obs.pos[at,i]-obs.pos[at,j]
                    n = raw_d/np.linalg.norm(raw_d,axis=1)[:,None]
                    blocks.append(normal_dynamics(clean,rr,f,i,j,n,calibration,base,pred))
                x = np.concatenate(blocks)
                assert len(x)==len(origins) and np.isfinite(x).all()
                if previous is not None:
                    np.testing.assert_allclose(x[:,0],previous,rtol=1e-9,atol=1e-8)
                np.savez_compressed(DIR/f'{held}__{tag}__{role}.npz',x=x.astype(np.float32),names=NAMES)
        print(tag,'fold-local normal dynamics prepared',flush=True)


if __name__=='__main__':
    prepare()
