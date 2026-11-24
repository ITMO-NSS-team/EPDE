from adequacy_context import *
import adequacy_fit as A

def main():
    rows=[]
    for key in ('stress_ode_positive_seed1283','stress_ode_positive_seed1334','stress_ode_positive_seed1456','stress_ode_positive_seed1016'):
        d=read(PREVIOUS/'signal_frozen'/(key+'.json'))
        with np.load(PREVIOUS/'signal_frozen'/(key+'.npz')) as z:y=z['raw']
        screens,choices=A.analyze(y,d['cv_info']);rows.append(dict(key=key,old_p=d['cv_info']['selected_p'],screens=screens,choices=choices))
    dump(ROOT/'development.json',dict(status='COMPLETE',role='Archived development only; no validation claim.',rows=rows,source_sha256=src('adequacy_fit.py','development.py')))
    for r in rows:print(r['key'],'old',r['old_p'],'choices',{k:v['selected_p'] for k,v in r['choices'].items()},'screens',[(x['p'],round(x['pvalue'],5),round(x['limit_lr'],3)) for x in r['screens']],flush=True)

if __name__=='__main__':main()
