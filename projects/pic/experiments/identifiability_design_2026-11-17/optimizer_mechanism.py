from research_context import *

def main():
    key='ode_seed52_bootstrap112';d=read(ROOT/'noise_inputs'/(key+'.json'));repair=read(ROOT/'repair_characterization'/(key+'.json'))
    with np.load(ROOT/'noise_inputs'/(key+'.npz')) as z:y=z['field']
    cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale]
    lo=np.array([-F.P.BOUND,-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([F.P.BOUND,20.,F.P.BOUND,F.P.BOUND]);rows=[]
    def exact(u):
        t,jac=F.P.inverse(u,1);v,g,_=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);return v,g*jac
    starts=[(f'band_start_{i}',r['transformed']) for i,r in enumerate(d['fit_info']['fits'])]+[('old_final',d['fit_info']['transformed']),('repaired',repair['info']['transformed'])]
    for name,uu in starts:
        u=np.array(uu);v,g=exact(u);t,jac=F.P.inverse(u,1);bv,bg=F.A.profile_score(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:],F.A.covariance_band);bg*=jac
        # Diagnostic finite differences only at moderate-rho points, away from bounds.
        fd=[]
        if max(abs(u[[0,2,3]]))<F.P.BOUND-.001:
            for j in range(4):
                step=np.zeros(4);step[j]=1e-5;fd.append((exact(u+step)[0]-exact(u-step)[0])/2e-5)
        row=dict(start=name,exact_objective=v,band_objective=bv,exact_gradient=g,band_gradient=bg,projected_gradient=float(max(abs(u-np.clip(u-g,lo,hi)))),finite_difference_gradient=fd,finite_difference_max_error=float(max(abs(np.array(fd)-g))) if fd else None)
        rows.append(row);print(name,'pg',row['projected_gradient'],'band_diff',float(max(abs(bg-g))),'FD',row['finite_difference_max_error'],flush=True)
    dump(ROOT/'optimizer_mechanism.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('optimizer_mechanism.py'),input_sha256={str(ROOT/'noise_inputs'/(key+'.json')):sha(ROOT/'noise_inputs'/(key+'.json')),str(ROOT/'repair_characterization'/(key+'.json')):sha(ROOT/'repair_characterization'/(key+'.json'))}))

if __name__=='__main__':main()
